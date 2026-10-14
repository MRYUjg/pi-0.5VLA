import dataclasses,pathlib
import flax.nnx as nnx
from openpi.models import pi0_config
from openpi.training import config,optimizer,weight_loaders
from openpi.shared import nnx_utils
VERSION=pathlib.Path(__file__).resolve().parents[1]
ROOT=VERSION.parents[1]
REPO=ROOT.parents[1]
COLLECTION=REPO/'agent-work-space/2-task-collection'
EXPORT=COLLECTION/'lerobot_exports/coverage_lerobot_r1'
TRAIN_ID='local/eco65_zhixing_can_coverage_r1_train'
VAL_ID='local/eco65_zhixing_can_coverage_r1_validation'
BASE=pathlib.Path('/openpi_assets/openpi-assets/checkpoints/pi05_libero/params')
def make(out):
    model=pi0_config.Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    return config.TrainConfig(name=VERSION.name,exp_name=out.name,model=model,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(BASE)),
        data=config.LeRobotLiberoDataConfig(repo_id=TRAIN_ID,assets=config.AssetsConfig(assets_dir=str(out/'assets'),asset_id=TRAIN_ID),base_config=config.DataConfig(prompt_from_task=True),extra_delta_transform=False),
        freeze_filter=nnx.Not(nnx_utils.PathRegex('.*lora.*')),batch_size=1,num_workers=0,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=10,peak_lr=1e-4,decay_steps=300,decay_lr=2e-5),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),ema_decay=None,num_train_steps=300,
        checkpoint_base_dir=str(out/'checkpoints'),assets_base_dir=str(out/'assets'),wandb_enabled=False)
