import gc
import torch
from typing import Dict, Union, Tuple
import torch.nn.functional as F
import torch.nn as nn
from quant.quant_layer import QuantLayer
from quant.quant_model import QuantModel
from models.quant_ldm_blocks import BaseQuantBlock, QuantResBlock, QuantBasicTransformerBlock
from quant.utils import get_op_name
import logging
import pdb

logger = logging.getLogger(__name__)

@torch.no_grad()
def save_inout(model: QuantModel, 
               layer: Union[QuantLayer, BaseQuantBlock], 
               cali_data: Tuple[torch.Tensor], 
               asym: bool = False, 
               use_wq: bool = True, 
               use_aq: bool = False, 
               batch_size: int = 128, 
               keep_gpu: bool = True, 
               ) -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
    device = next(model.parameters()).device
    get_inout = GetLayerInpOut(model, layer, device, asym, use_wq, use_aq, keep_gpu=keep_gpu)
    cached_inputs, cached_outputs = None, None
    gc.collect()
    torch.cuda.empty_cache()

    in_save_device = 'cpu'
    out_save_device = 'cpu'
    
    for i in range(0, cali_data[0].size(0), batch_size):
        ipts, opts = get_inout(*(_[i: i + batch_size] for _ in cali_data))

        if cached_inputs is None:
            cached_inputs = tuple([] for _ in range(len(ipts)))
        if cached_outputs is None:
            cached_outputs = tuple([] for _ in range(len(opts)))

        for j in range(len(ipts)):
            cached_inputs[j].append(ipts[j].detach().to(in_save_device))

        # if keep_gpu:
        for j in range(len(opts)):
            cached_outputs[j].append(opts[j].detach().to(out_save_device))

    gc.collect()
    torch.cuda.empty_cache()
    cached_inputs = tuple((torch.cat([y for y in x]) for x in cached_inputs))
    cached_outputs = tuple((torch.cat([y for y in x]) for x in cached_outputs))
    # cached_inputs = tuple(x.to(device) for x in cached_inputs)
    if keep_gpu:
        cached_inputs = tuple(x.to(device) for x in cached_inputs)
        cached_outputs = tuple(x.to(device) for x in cached_outputs)

    print([x.shape for x in cached_inputs])
      
    for i, x in enumerate(cached_inputs):
        if '0' in str(x.device) or 'cpu' in str(x.device):
            logger.info(f'input {i} shape: {x.shape}')
    for i, x in enumerate(cached_outputs):
        if '0' in str(x.device) or 'cpu' in str(x.device):
            logger.info(f'output {i} shape: {x.shape}')
    gc.collect()
    torch.cuda.empty_cache()
    if len(cached_outputs) == 1:
        return cached_inputs, cached_outputs[0]
    return cached_inputs, cached_outputs  


class StopForwardException(Exception):
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, 
                 store_input: bool = False, 
                 store_output: bool = False, 
                 stop_forward: bool = False
                 ) -> None:
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, 
                 module: nn.Module, 
                 input_batch: Tuple[Union[torch.Tensor, int]], # ResBlock or ResnetBlock has an integer input split
                 kwargs: Dict[str, Union[torch.Tensor, int]],
                 output_batch: Union[torch.Tensor, Tuple[torch.Tensor]]
                 ) -> None:
        if self.store_input:
            self.input_store = input_batch
            if isinstance(input_batch[-1], int):
                self.input_store = input_batch[:-1]
            if isinstance(module, QuantBasicTransformerBlock): # in order to capture context
                self.input_store += tuple(kwargs.values())
                
        if self.store_output:
            # Handle cases where output might be a tuple but we only want the first element
            if isinstance(module, QuantResBlock) and len(output_batch) != 1:
                self.output_store = output_batch[0]
            else:
                self.output_store = output_batch

        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(self, model: QuantModel, 
                 layer: Union[QuantLayer, BaseQuantBlock],
                 device: torch.device, 
                 asym: bool = False,
                 use_wq: bool = True,
                 use_aq: bool = False,
                 keep_gpu: bool = True
                 ) -> None:
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.use_wq = use_wq
        self.use_aq = use_aq
        self.keep_gpu = keep_gpu
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, 
                 xs: torch.Tensor, 
                 ts: torch.Tensor, 
                 cs: torch.Tensor = None
                 ) -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        self.model.eval()
        self.model.set_quant_state(False, False)

        # if self.keep_gpu == False:
        #     self.data_saver.store_output = False

        handle = self.layer.register_forward_hook(self.data_saver, with_kwargs=True)
        with torch.no_grad():
            try:
                inputs = [xs.to(self.device), ts.to(self.device)]
                inputs = inputs + [cs.to(self.device)] if cs is not None else inputs
                _ = self.model(*inputs)
            except StopForwardException:
                pass

            if self.asym:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(use_wq=self.use_wq, use_aq=self.use_aq)
                try:
                    inputs = [xs.to(self.device), ts.to(self.device)]
                    inputs = inputs + [cs.to(self.device)] if cs is not None else inputs
                    _ = self.model(*inputs)
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()
        
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(self.use_wq, self.use_aq)

        input_stores = tuple(x.detach() for x in self.data_saver.input_store if torch.is_tensor(x))
        if self.data_saver.output_store == None:
            return input_stores, tuple()
        elif isinstance(self.data_saver.output_store, torch.Tensor):
            output_stores = tuple([self.data_saver.output_store.detach()])
            return input_stores, output_stores
        output_stores = tuple(x.detach() for x in self.data_saver.output_store)
        
        return input_stores, output_stores
