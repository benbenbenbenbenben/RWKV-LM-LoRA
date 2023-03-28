from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.cli import LightningCLI
import numpy as np
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import torch
import warnings, datetime, os, importlib
from torch.utils.data import DataLoader

class MyClassifier(LightningModule):
    pass


class MyDataModule(LightningDataModule):
    def __init__(
        self,
        load_model: str = "",
        wandb: str = "",
        proj_dir: str = "out",
        random_seed: int = -1,
        data_file: str = "",
        data_type: str = "utf-8",
        vocab_size: int = 0,
        ctx_len: int = 1024,
        epoch_steps: int = 1000,
        epoch_count: int = 500,
        epoch_begin: int = 0,
        epoch_save: int = 5,
        micro_bsz: int = 12,
        n_layer: int = 6,
        n_embd: int = 512,
        dim_att: int = 0,
        dim_ffn: int = 0,
        pre_ffn: int = 0,
        head_qk: int = 0,
        tiny_att_dim: int = 0,
        tiny_att_layer: int = -999,
        lr_init: float = 6e-4,
        lr_final: float = 1e-5,
        warmup_steps: int = 0,
        beta1: float = 0.9,
        beta2: float = 0.99,
        adam_eps: float = 1e-8,
        grad_cp: int = 0,
        my_pile_version: int = 1,
        my_pile_stage: int = 0,
        my_pile_shift: int = -1,
        my_pile_edecay: int = 0,
        layerwise_lr: int = 1,
        ds_bucket_mb: int = 200,
        my_img_version: str = "",
        my_img_size: int = 0,
        my_img_bit: int = 0,
        my_img_clip: str = "x",
        my_img_clip_scale: float = 1,
        my_img_l1_scale: float = 0,
        my_img_encoder: str = "x",
        my_sample_len: int = 0,
        my_ffn_shift: int = 1,
        my_att_shift: int = 1,
        my_pos_emb: int = 0,
        load_partial: int = 0,
        magic_prime: int = 0,
        my_qa_mask: int = 0,
        my_testing: str = "",
        lora: bool = False,
        lora_load: str = "",
        lora_r: int = 8,
        lora_alpha: float = 32,
        lora_dropout: float = 0.01,
        lora_parts: str = "att,ln,time",
        strategy: str = None,
    ):
        super().__init__()
        self.load_model = load_model
        self.wandb = wandb
        self.proj_dir = proj_dir
        self.random_seed = random_seed
        self.data_file = data_file
        self.data_type = data_type
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.epoch_steps = epoch_steps
        self.epoch_count = epoch_count
        self.epoch_begin = epoch_begin
        self.epoch_save = epoch_save
        self.micro_bsz = micro_bsz
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.dim_att = dim_att
        self.dim_ffn = dim_ffn
        self.pre_ffn = pre_ffn
        self.head_qk = head_qk
        self.tiny_att_dim = tiny_att_dim
        self.tiny_att_layer = tiny_att_layer
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
        self.grad_cp = grad_cp
        self.my_pile_version = my_pile_version
        self.my_pile_stage = my_pile_stage
        self.my_pile_shift = my_pile_shift
        self.my_pile_edecay = my_pile_edecay
        self.layerwise_lr = layerwise_lr
        self.ds_bucket_mb = ds_bucket_mb
        self.my_img_version = my_img_version
        self.my_img_size = my_img_size
        self.my_img_bit = my_img_bit
        self.my_img_clip = my_img_clip
        self.my_img_clip_scale = my_img_clip_scale
        self.my_img_l1_scale = my_img_l1_scale
        self.my_img_encoder = my_img_encoder
        self.my_sample_len = my_sample_len
        self.my_ffn_shift = my_ffn_shift
        self.my_att_shift = my_att_shift
        self.my_pos_emb = my_pos_emb
        self.load_partial = load_partial
        self.magic_prime = magic_prime
        self.my_qa_mask = my_qa_mask
        self.my_testing = my_testing
        self.lora = lora
        self.lora_load = lora_load
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_parts = lora_parts
        self.strategy = strategy

        if self.strategy is not None and "deepspeed" in self.strategy:
            import deepspeed

        if self.random_seed >= 0:
            print(f"########## WARNING: GLOBAL SEED {self.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
            seed_everything(self.random_seed)
        
        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
        warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

        self.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        self.enable_checkpointing = False
        self.replace_sampler_ddp = False
        self.logger = False
        self.gradient_clip_val = 1.0
        self.num_sanity_val_steps = 0
        self.check_val_every_n_epoch = int(1e20)
        self.log_every_n_steps = int(1e20)
        self.max_epochs = -1  # continue forever
        self.betas = (self.beta1, self.beta2)
        self.real_bsz = int(self.num_nodes) * int(self.devices) * self.micro_bsz
        os.environ["RWKV_T_MAX"] = str(self.ctx_len)
        os.environ["RWKV_MY_TESTING"] = self.my_testing
        if self.dim_att <= 0:
            self.dim_att = self.n_embd
        if self.dim_ffn <= 0:
            self.dim_ffn = self.n_embd * 4
        
        if self.data_type == "wds_img":
            self.run_name = f"v{self.my_img_version}-{self.my_img_size}-{self.my_img_bit}bit-{self.my_img_clip}x{self.my_img_clip_scale}"
            self.proj_dir = f"{self.proj_dir}-{self.run_name}"
        else:
            self.run_name = f"{self.vocab_size} ctx{self.ctx_len} L{self.n_layer} D{self.n_embd}"
        if not os.path.exists(self.proj_dir):
            os.makedirs(self.proj_dir)
        if self.my_pile_stage > 0:
            magic_prime_bak = self.magic_prime

            if self.my_pile_version == 1:
                if self.ctx_len == 1024:
                    self.magic_prime = 324331313
                    self.epoch_count = 8043
                elif self.ctx_len == 2048:
                    self.magic_prime = 162165671
                    self.epoch_count = 4021
                elif self.ctx_len == 4096:
                    self.magic_prime = 81082817
                    self.epoch_count = 2010
                elif self.ctx_len == 8192:
                    self.magic_prime = 40541399
                    self.epoch_count = 1005
            else:
                if self.ctx_len == 4096:
                    self.magic_prime = 423736637
                    self.epoch_count = 10508
                elif self.ctx_len == 8192:
                    self.magic_prime = 211868243
                    self.epoch_count = 5253
            if self.my_pile_shift < 0:
                self.my_pile_shift = 0

            if magic_prime_bak > 0:
                self.magic_prime = magic_prime_bak

            self.epoch_steps = 40320 // self.real_bsz
            assert self.epoch_steps * self.real_bsz == 40320
            if self.my_pile_stage == 2:
                assert self.lr_final == self.lr_init
            if self.my_pile_stage >= 2:  # find latest saved model
                list_p = []
                for p in os.listdir(self.proj_dir):
                    if p.startswith("rwkv") and p.endswith(".pth"):
                        p = ((p.split("-"))[1].split("."))[0]
                        if p == "init":
                            p = -1
                        else:
                            p = int(p)
                        list_p += [p]
                list_p.sort()
                max_p = list_p[-1]
                if len(list_p) > 1:
                    self.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
                if max_p == -1:
                    self.load_model = f"{self.proj_dir}/rwkv-init.pth"
                else:
                    self.load_model = f"{self.proj_dir}/rwkv-{max_p}.pth"
                    if self.my_pile_stage == 2:
                        self.warmup_steps = 10
                    else:
                        self.warmup_steps = 30
                self.epoch_begin = max_p + 1

        samples_per_epoch = self.epoch_steps * self.real_bsz
        tokens_per_epoch = samples_per_epoch * self.ctx_len
        rank_zero_info(
            f"""
############################################################################
#
# RWKV-4 {self.precision.upper()} on {self.num_nodes}x{self.devices} {self.accelerator.upper()}, bsz {self.num_nodes}x{self.devices}x{self.micro_bsz}={self.real_bsz}, {self.strategy} {'with grad_cp' if self.grad_cp > 0 else ''}
#
# Data = {self.data_file} ({self.data_type}), ProjDir = {self.proj_dir}
#
# Epoch = {self.epoch_begin} to {self.epoch_begin + self.epoch_count - 1} (will continue afterwards), save every {self.epoch_save} epoch
#
# Each "epoch" = {self.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {self.n_layer} n_layer, {self.n_embd} n_embd, {self.ctx_len} ctx_len
# LoRA = {f'enabled, {self.lora_r} r, {self.lora_alpha} alpha, {self.lora_dropout} dropout, on {self.lora_parts}' if self.lora else 'disabled'}
#
# Adam = lr {self.lr_init} to {self.lr_final}, warmup {self.warmup_steps} steps, beta {self.betas}, eps {self.adam_eps}
#
# Found torch {torch.__version__}, recommend 1.13.1+cu117 or newer
# Found deepspeed {deepspeed.__version__ if importlib.util.find_spec('deepspeed') else 'None'}, recommend 0.7.0 (faster than newer versions)
# Found pytorch_lightning {pl.__version__}, recommend 1.9.1 or newer
#
############################################################################
"""
        )
        rank_zero_info(str(vars(self)) + "\n")

        assert self.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "wds_img", "uint16"]

        if self.lr_final == 0 or self.lr_init == 0:
            rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

        assert self.precision in ["fp32", "tf32", "fp16", "bf16"]
        os.environ["RWKV_FLOAT_MODE"] = self.precision
        if self.precision == "fp32":
            for i in range(10):
                rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
        if self.precision == "fp16":
            rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

        os.environ["RWKV_JIT_ON"] = "1"
        if "deepspeed_stage_3" in self.strategy:
            os.environ["RWKV_JIT_ON"] = "0"
        if self.lora and self.grad_cp == 1:
            print('!!!!! LoRA Warning: Gradient Checkpointing requires JIT off, disabling it')
            os.environ["RWKV_JIT_ON"] = "0"

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if self.precision == "fp32":
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        else:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

        if "32" in self.precision:
            self.precision = 32
        elif self.precision == "fp16":
            self.precision = 16
        else:
            self.precision = "bf16"
        from src.trainer import train_callback, generate_init_weight
        from src.dataset import MyDataset

        train_data = MyDataset(self)
        self.vocab_size = train_data.vocab_size

        if self.data_type == 'wds_img':
            from src.model_img import RWKV_IMG
            assert self.lora, "LoRA not yet supported for RWKV_IMG"
            model = RWKV_IMG(self)
        else:
            from src.model import RWKV, LORA_CONFIG, LoraLinear
            if self.lora:
                assert self.lora_r > 0, "LoRA should have its `r` > 0"
                LORA_CONFIG["r"] = self.lora_r
                LORA_CONFIG["alpha"] = self.lora_alpha
                LORA_CONFIG["dropout"] = self.lora_dropout
                LORA_CONFIG["parts"] = set(str(self.lora_parts).split(','))
                enable_time_finetune = 'time' in LORA_CONFIG["parts"]
                enable_ln_finetune = 'ln' in LORA_CONFIG["parts"]
            model = RWKV(self)
            # only train lora parameters
            if self.lora:
                model.requires_grad_(False)
                for name, module in model.named_modules():
                    # have to check param name since it may have been wrapped by torchscript
                    if 'lora_A' in set(n for n, _ in module.named_parameters()):
                        print(f'  LoRA training {name}')
                        for pname, param in module.named_parameters():
                            param.requires_grad = 'lora_' in pname
                    elif ((enable_time_finetune and '.time_' in name)
                            or (enable_ln_finetune and '.ln' in name)):
                        print(f'  LoRA additionally training {name}')
                        for param in module.parameters():
                            param.requires_grad = True

        if len(self.load_model) == 0 or self.my_pile_stage == 1:  # shall we build the initial weights?
            init_weight_name = f"{self.proj_dir}/rwkv-init.pth"
            generate_init_weight(model, init_weight_name)  # save initial weights
            self.load_model = init_weight_name

        rank_zero_info(f"########## Loading {self.load_model}... ##########")
        try:
            load_dict = torch.load(self.load_model, map_location="cpu")
        except:
            rank_zero_info(f"Bad checkpoint {self.load_model}")
            if self.my_pile_stage >= 2:  # try again using another checkpoint
                max_p = self.my_pile_prev_p
                if max_p == -1:
                    self.load_model = f"{self.proj_dir}/rwkv-init.pth"
                else:
                    self.load_model = f"{self.proj_dir}/rwkv-{max_p}.pth"
                self.epoch_begin = max_p + 1
                rank_zero_info(f"Trying {self.load_model}")
                load_dict = torch.load(self.load_model, map_location="cpu")

        if self.load_partial == 1:
            load_keys = load_dict.keys()
            for k in model.state_dict():
                if k not in load_keys:
                    load_dict[k] = model.state_dict()[k]
        # If using LoRA, the LoRA keys might be missing in the original model
        model.load_state_dict(load_dict, strict=(not self.lora))
        if os.path.isfile(self.lora_load):
            model.load_state_dict(torch.load(self.lora_load, map_location="cpu"),
                                strict=False)

        # trainer = LightningCLI(
        #     self,
        #     callbacks=[train_callback(self)],
        # )

        if self.global_rank == 0:
            for n in model.state_dict():
                shape = model.state_dict()[n].shape
                shape = [i for i in shape if i != 1]
                if len(shape) > 1:
                    print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {n}")
                else:
                    print(f"{str(shape[0]).ljust(5)}       {n}")

        # if "deepspeed" in self.strategy:
        #     trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = self.ds_bucket_mb * 1000 * 1000
        #     trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = self.ds_bucket_mb * 1000 * 1000

        # must set shuffle=False, persistent_workers=False (because worker is in another thread)
        data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=self.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

if __name__ == "__main__":
    cli = LightningCLI(MyClassifier, MyDataModule, seed_everything_default=42)
    #result = cli.trainer.fit(cli.model, cli.datamodule)
