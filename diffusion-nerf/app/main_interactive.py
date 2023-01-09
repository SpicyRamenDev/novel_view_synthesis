# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

def my_get_modules_from_config(args):
    """Utility function to get the modules for training from the parsed config.
    """
    import torch
    from wisp.config_parser import load_mv_grid
    from wisp.models import Pipeline
    from wisp.models.grids import OctreeGrid
    from wisp.datasets import MultiviewDataset
    from wisp.datasets.transforms.ray_sampler import SampleRays
    from my_ray_sampler import MySampleRays
    from my_nerf import MyNeuralRadianceField
    from my_packed_rf_tracer import MyPackedRFTracer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = SampleRays(args.num_rays_sampled_per_img)
    if args.use_gt:
        train_dataset = MultiviewDataset(**vars(args), transform=transform)
    else:
        train_dataset = None
    grid = load_mv_grid(args, train_dataset)
    nef = MyNeuralRadianceField(grid, **vars(args))
    tracer = MyPackedRFTracer(**vars(args))
    pipeline = Pipeline(nef, tracer)
    pipeline = pipeline.to(device)

    return pipeline, train_dataset, device

def parse_args():
    from wisp.config_parser import parse_options, argparse_to_str

    # Usual boilerplate
    parser = parse_options(return_parser=True)
    app_utils.add_log_level_flag(parser)

    # Add custom args if needed for app
    app_group = parser.add_argument_group('app')

    my_app_group = parser.add_argument_group('my_app')
    my_app_group.add_argument('--use-gt', action='store_true', default=False,
                              help='Use gt views.')
    my_app_group.add_argument('--no-gt', action='store_false', dest='use_gt',
                              help='Do not use gt views.')

    my_diffusion_group = parser.add_argument_group('my_diffusion')
    my_diffusion_group.add_argument('--repo-id', type=str, default='stabilityai/stable-diffusion-2-1-base',
                                    help='The repo id for the diffusion model to use.')
    my_diffusion_group.add_argument('--diffusion-type', type=str, default='StableDiffusion',
                                    help='The diffusion model to use.')
    my_diffusion_group.add_argument('--prompt', type=str, default='a 3D render of an apple',
                                    help='The prompt to use for the diffusion model.')
    my_diffusion_group.add_argument('--negative-prompt', type=str, default='',
                                    help='The negative prompt to use for the diffusion model.')
    my_diffusion_group.add_argument('--diffusion-bg-color', type=str, default='noise',
                                    help='Background color for diffusion.')
    my_diffusion_group.add_argument('--guidance-scale', type=int, default=100,
                                    help='The guidance scale.')
    my_app_group.add_argument('--use-view-prompt', action='store_true', default=True,
                              help='Use view prompt.')
    my_app_group.add_argument('--no-view-prompt', action='store_false', dest='use_view_prompt',
                              help='Do not use view prompt.')

    my_nef_group = parser.add_argument_group('my_nef')
    my_nef_group.add_argument('--pos-embedder', type=str, default='positional',
                              help='Pos embedder type.')
    my_nef_group.add_argument('--view-embedder', type=str, default='spherical',
                              help='View embedder type.')
    my_nef_group.add_argument('--blob-scale', type=float, default=5.0,
                              help='Scale applied to the gaussian blob of density.')
    my_nef_group.add_argument('--blob-width', type=float, default=0.2,
                              help='width of the gaussian blob of density.')
    my_nef_group.add_argument('--use-blob', action='store_true', default=False,
                              help='Use the gaussian blob of density.')
    my_nef_group.add_argument('--no-blob', action='store_false', dest='use_blob',
                              help='Do not use the gaussian blob of density.')
    my_nef_group.add_argument('--bottleneck_dim', type=int, default=8,
                              help='Dimension of the bottleneck.')

    my_optimizer_group = parser.add_argument_group('my_optimizer')
    my_optimizer_group.add_argument('--orientation-loss', type=float, default=0.1,
                                    help='Orientation regularizer loss multiplier.')
    my_optimizer_group.add_argument('--predicted-normal-loss', type=float, default=0.2,
                                    help='Predicted normal regularizer loss multiplier.')
    my_optimizer_group.add_argument('--opacity-loss', type=float, default=0.001,
                                    help='Opacity loss multiplier.')
    my_optimizer_group.add_argument('--entropy-loss', type=float, default=0.002,
                                    help='Entropy loss multiplier.')
    my_optimizer_group.add_argument('--diffusion-loss', type=float, default=1.0,
                                    help='Diffusion loss multiplier.')
    my_optimizer_group.add_argument('--num-novel-views-per-gt', type=int, default=0,
                                    help='Opacity loss multiplier.')
    my_optimizer_group.add_argument('--num-novel-views-base', type=int, default=100,
                                    help='Opacity loss multiplier.')
    my_optimizer_group.add_argument('--albedo-steps', type=int, default=1000,
                                    help='Number of albedo steps.')
    my_optimizer_group.add_argument('--warmup-iterations', type=int, default=3000,
                                    help='Warmup iterations.')
    my_optimizer_group.add_argument('--init-lr', type=float, default=1e-5,
                                    help='Init iterations.')
    my_optimizer_group.add_argument('--end-lr', type=float, default=1e-2,
                                    help='End iterations.')
    my_optimizer_group.add_argument('--reg-warmup-iterations', type=int, default=3000,
                                    help='Reg warmup iterations.')
    my_optimizer_group.add_argument('--reg-init-lr', type=float, default=1e-5,
                                    help='Reg init iterations.')
    my_optimizer_group.add_argument('--coarse-ratio', type=float, default=0.5,
                                    help='Coarse ratio of iterations.')

    my_camera_group = parser.add_argument_group('my_camera')
    my_camera_group.add_argument('--azimuth-range', nargs=2, type=float, default=[0, 360],
                                 help='Azimuth range of the camera.')
    my_camera_group.add_argument('--polar-range', nargs=2, type=float, default=[0, 100],
                                 help='Polar range of the camera.')
    my_camera_group.add_argument('--camera-distance-range', nargs=2, type=float, default=[3.5, 4.5],
                                 help='Range of the camera distance.')
    my_camera_group.add_argument('--camera-angle-range', nargs=2, type=float, default=[25, 42],
                                 help='Range of the camera angle in degrees.')
    my_camera_group.add_argument('--focal-length-multiplier-range', nargs=2, type=float, default=[0.7, 1.35],
                                 help='Range of the focal length multiplier.')
    my_camera_group.add_argument('--camera-offset', type=float, default=0.1,
                                 help='Offset of the camera from its position.')
    my_camera_group.add_argument('--camera-up-std', type=float, default=0.02,
                                 help='Std of the camera up vector.')
    my_camera_group.add_argument('--look-at-std', type=float, default=0.2,
                                 help='Std of the look up vector.')
    my_camera_group.add_argument('--light-distance-range', nargs=2, type=float, default=[0.8, 1.5],
                                 help='Std of the look up vector.')
    my_camera_group.add_argument('--coarse-resolution', type=int, default=64,
                                 help='Coarse resolution.')
    my_camera_group.add_argument('--fine-resolution', type=int, default=512,
                                 help='Fine resolution.')

    args, args_str = argparse_to_str(parser)
    return args, args_str


def create_trainer(args, scene_state):
    """ Create the trainer according to config args """
    from stable_diffusion import StableDiffusion
    from wisp.config_parser import get_modules_from_config, get_optimizer_from_config
    pipeline, train_dataset, device = my_get_modules_from_config(args)
    optim_cls, optim_params = get_optimizer_from_config(args)
    diffusion = StableDiffusion(device=device, repo_id="stabilityai/stable-diffusion-2-1-base")
    trainer = MyTrainerSD(pipeline, train_dataset, args.epochs, args.batch_size,
                          optim_cls, args.lr, args.weight_decay,
                          args.grid_lr_weight, optim_params, args.log_dir, device,
                          diffusion=diffusion,
                          exp_name=args.exp_name, info=args_str, extra_args=vars(args),
                          render_tb_every=args.render_tb_every, save_every=args.save_every,
                          scene_state=scene_state, using_wandb=using_wandb)
    return trainer


def create_app(scene_state, trainer):
    """ Create the interactive app running the renderer & trainer """
    from my_app import MyApp
    from wisp.renderer.app.optimization_app import OptimizationApp
    scene_state.renderer.device = trainer.device  # Use same device for trainer and renderer
    interactive_app = MyApp(wisp_state=scene_state, background_task=trainer.iterate, window_name="My App")
    #interactive_app = OptimizationApp(wisp_state=scene_state, trainer_step_func=trainer.iterate, experiment_name="Opt App")
    return interactive_app


if __name__ == "__main__":
    # Must be called before any torch operations take place
    from app.cuda_guard import setup_cuda_context
    setup_cuda_context()

    import os
    import app.app_utils as app_utils
    import logging as log
    from wisp.framework import WispState

    import wandb

    from wisp.trainers import *

    # Register any newly added user classes before running the config parser
    # Registration ensures the config parser knows about these classes and is able to dynamically create them.
    from wisp.config_parser import register_class
    from my_nerf import MyNeuralRadianceField
    from my_trainer import MyTrainer
    from my_trainer_sd import MyTrainerSD
    from my_packed_rf_tracer import MyPackedRFTracer
    register_class(MyNeuralRadianceField, 'MyNeuralRadianceField')
    register_class(MyTrainer, 'MyTrainer')
    register_class(MyTrainerSD, 'MyTrainerSD')
    register_class(MyPackedRFTracer, 'MyPackedRFTracer')
    from wisp.renderer.core.api import register_neural_field_type
    from wisp.renderer.core.renderers import NeuralRadianceFieldPackedRenderer
    register_neural_field_type(neural_field_type=MyNeuralRadianceField,
                               tracer_type=MyPackedRFTracer,
                               renderer_type=NeuralRadianceFieldPackedRenderer)

    # Parse config yaml and cli args
    args, args_str = parse_args()

    using_wandb = args.wandb_project is not None
    if using_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name if args.wandb_run_name is None else args.wandb_run_name,
            entity=args.wandb_entity,
            job_type="validate" if args.valid_only else "train",
            config=vars(args),
            sync_tensorboard=True
        )

    app_utils.default_log_setup(args.log_level)

    # Create the state object, shared by all wisp components
    scene_state = WispState()

    # Create the trainer
    trainer = create_trainer(args, scene_state)

    if not os.environ.get('WISP_HEADLESS') == '1':
        interactive_app = create_app(scene_state, trainer)
        interactive_app.run()
    else:
        log.info("Running headless. For the app, set WISP_HEADLESS=0")
        if args.valid_only:
            trainer.validate()
        else:
            trainer.train()

    if args.trainer_type in ["MultiviewTrainer", "MYTrainer"] and using_wandb and args.wandb_viz_nerf_angles != 0:
        print('Final render')
        trainer.render_final_view(
            num_angles=args.wandb_viz_nerf_angles,
            camera_distance=args.wandb_viz_nerf_distance
        )

    if using_wandb:
        wandb.finish()
