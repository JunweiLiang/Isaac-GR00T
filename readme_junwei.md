+ 微调Gr00T N1.6
    + (04/17/2026) teleop采集数据(unitree 格式): `https://github.com/hkustgz-hw/humanoid_teleop_hkustgz-hw/blob/main/DATASET.md`
    + (04/17/2026) 处理后数据(lerobot v2和v3格式): `machine11: /mnt/nvme2/junweil/lerobot_wbc_datasets_v3+v2_5tasks+5singletask.tgz`
    ```
        数据可以放到huggingface默认的路径，比如
        ~/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground/
        训练就可以用 --repo-id junweiliang/wbc_pick_up_object_from_ground
    ```
    + (04/17/2026) 模型: `machine11: /mnt/nvme2/junweil/models`
    ```
        (base) junweil@ai-precog-machine11:/mnt/nvme2/junweil/models$ tree -L 2
            .
            ├── my_wbc_move_box
            │   └── checkpoint-10000
            ├── my_wbc_pick_up_object_from_ground_bs64_s10k
            │   └── checkpoint-10000
            └── mygpu3_wbc_close_washer_door
                └── checkpoint-10000
    ```
    + (04/17/2026) finetune states/action配置：`./my_configs/g1_dex3_gripper_homie.py`
    + open-loop 评测:
    ```

        junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T$ uv run python gr00t/eval/open_loop_eval.py     --dataset-path ~/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground_val0.1/  --embodiment-tag NEW_EMBODIMENT     --model-path experiments/my_wbc_pick_up_object_from_ground_bs64_s10k/checkpoint-10000/     --traj-ids 0 1 2 3 4 5     --action-horizon 16 --save-plot-path experiments/my_wbc_pick_up_object_from_ground_bs64_s10k/open_loop_val_ep0-5.jpg --steps 1000

            INFO:root:Average MSE across all trajs: 0.002199870301410556
            INFO:root:Average MAE across all trajs: 0.01576397754251957
    ```
    + 实机测试前，先看看训练数据长什么样，尽量把环境布置一样
    ```
        # https://github.com/JunweiLiang/humanoid_teleop

        (tv) junweil@office-precognition:~/projects$ python ~/projects/humanoid_teleop/g1_realrobot/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py --repo-id junweiliang/wbc_pick_up_object_from_ground --episode-index 0

    ```
    + 详细记录，包括环境安装、数据处理、微调、open-loop评测
    ```
    [04/14/2026] # fork and install
        # https://github.com/JunweiLiang/Isaac-GR00T
        # office 已经安装了uv, CUDA 12.6 在根环境
        junweil@office-precognition:~/projects/wbc_manipulation$ git clone https://github.com/JunweiLiang/Isaac-GR00T

        junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T$ git submodule update --init --recursive

        junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T$ bash scripts/deployment/dgpu/install_deps.sh

        # 下载 N1.6 checkpoint

            $ pip install -U "huggingface_hub[cli]"
            (base) junweil@office-precognition:~/projects/wbc_manipulation$ hf download nvidia/GR00T-N1.6-3B --local-dir ./GR00T-N1.6-3B

        # gpu3 安装
            # 1. 安装uv

                curl -LsSf https://astral.sh/uv/install.sh | sh

            # 2. 安装cuda 12.6
                $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
                $ sudo dpkg -i cuda-keyring_1.1-1_all.deb
                $ sudo apt-get update
                $ sudo apt-get install -y cuda-toolkit-12-6 pybind11-dev libgmpxx4ldbl libgmp-dev

                $ vi ~/.bashrc
                export CUDA_HOME=/usr/local/cuda-12.6
                export PATH=$CUDA_HOME/bin:$PATH
                export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

            # 3. gr00t
                junweil@precognition-gpu3:~/projects/wbc_manipulation$ git clone https://github.com/JunweiLiang/Isaac-GR00T
                junweil@precognition-gpu3:~/projects/wbc_manipulation/Isaac-GR00T$ git submodule update --init --recursive

                # 要用清华源
                junweil@precognition-gpu3:~/projects/wbc_manipulation/Isaac-GR00T$ UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple" UV_CONCURRENT_DOWNLOADS=10 UV_HTTP_TIMEOUT=60 bash scripts/deployment/dgpu/install_deps.sh



        # 2x4090 48GB 测试
            # 单个任务训练

                # 多卡必须torchrun

                junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T$ uv run torchrun --nproc_per_node=2 gr00t/experiment/launch_finetune.py      --base-model-path ../GR00T-N1.6-3B      --dataset-path ~/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground      --embodiment-tag NEW_EMBODIMENT      --modality-config-path my_configs/g1_dex3_gripper_homie.py      --save-total-limit 5      --learning_rate 1e-4      --save-steps 2000      --max-steps 10000      --use-wandb      --warmup_ratio 0.05      --weight_decay 1e-5      --global-batch-size 32      --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08      --dataloader-num-workers 6      --output-dir experiments/my_wbc_pick_up_object_from_ground

                # 需要一个wandb账号
                    # https://wandb.ai/authorize?signup=true&ref=models
                    # 需要API key  ~/Desktop/github_projects/wandb_api_key.txt

                # 2x4090, bs=32, s=10k 需要3小时wbc_pick_up_object_from_ground
                    # bs=64,s=10k 需要4小时move_box
                    # bs=128 OOM
                    # dataloader-num-workers 4 比 8要更快 (32 thread CPU, 所以除以8)
                    # GPU温度44度，260/450w, 所以效率比较差
                    # CPU 90%, 内存33GB

                # 看training log: https://wandb.ai/home

                # open-loop 评测

                    # 训练集, 挑10个ep, 看平均的MSE, MAE

                        junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T$ uv run python gr00t/eval/open_loop_eval.py     --dataset-path ~/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground/  --embodiment-tag NEW_EMBODIMENT     --model-path experiments/my_wbc_pick_up_object_from_ground/checkpoint-10000/     --traj-ids 0 1 2 3 4 5 6 7 8 9     --action-horizon 16 --save-plot-path experiments/my_wbc_pick_up_object_from_ground/open_loop_train_ep0-10.jpg

                        INFO:root:Average MSE across all trajs: 0.00018743000691756606
                        INFO:root:Average MAE across all trajs: 0.004905599635094404

                    # validation set, 0-5 ep

                        junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T$ uv run python gr00t/eval/open_loop_eval.py     --dataset-path ~/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground_val0.1/  --embodiment-tag NEW_EMBODIMENT     --model-path experiments/my_wbc_pick_up_object_from_ground/checkpoint-10000/     --traj-ids 0 1 2 3 4 5     --action-horizon 16 --save-plot-path experiments/my_wbc_pick_up_object_from_ground/open_loop_val_ep0-5.jpg --steps 1000

                        INFO:root:Average MSE across all trajs: 0.0024767746217548847
                        INFO:root:Average MAE across all trajs: 0.01650149933993816


                    # 可视化一下, 还有看上面的关节推理plot

                    (tv) junweil@office-precognition:~/projects$ python ~/projects/humanoid_teleop/g1_realrobot/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py --repo-id junweiliang/wbc_pick_up_object_from_ground --episode-index 3

        # gr00t 官方wbc 的finetune config
            export NUM_GPUS=8

            torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
                gr00t/experiment/launch_finetune.py \
                --base_model_path  nvidia/GR00T-N1.6-3B \
                --dataset_path examples/GR00T-WholeBodyControl/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/unitree_g1.LMPnPAppleToPlateDC \
                --embodiment_tag UNITREE_G1 \
                --num_gpus $NUM_GPUS \
                --output_dir /tmp/g1_finetune \
                --save_total_limit 5 \
                --max_steps 10000 \
                --warmup_ratio 0.05 \
                --weight_decay 1e-5 \
                --learning_rate 1e-4 \
                --use_wandb \
                --global_batch_size 1024 \
                --dataloader_num_workers 6 \
                --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

        # gpu3获取数据

            (tv) junweil@office-precognition:~/.cache/huggingface/lerobot$ scp lerobot_wbc_datasets_v3+v2_5tasks+5singletask.tgz  junweil@gpu3.precognition.team:~/

            # 数据放在
                (base) junweil@precognition-gpu3:~/projects/wbc_manipulation/junweiliang$ ls
                    wbc_5tasks                         wbc_move_and_open_pot              wbc_open_washer_door
                    wbc_5tasks_v3.0                    wbc_move_and_open_pot_v3.0         wbc_open_washer_door_v3.0
                    wbc_5tasks_val0.1                  wbc_move_and_open_pot_val0.1       wbc_open_washer_door_val0.1
                    wbc_5tasks_val0.1_v3.0             wbc_move_and_open_pot_val0.1_v3.0  wbc_open_washer_door_val0.1_v3.0
                    wbc_close_washer_door              wbc_move_box                       wbc_pick_up_object_from_ground
                    wbc_close_washer_door_v3.0         wbc_move_box_v3.0                  wbc_pick_up_object_from_ground_v3.0
                    wbc_close_washer_door_val0.1       wbc_move_box_val0.1                wbc_pick_up_object_from_ground_val0.1
                    wbc_close_washer_door_val0.1_v3.0  wbc_move_box_val0.1_v3.0           wbc_pick_up_object_from_ground_val0.1_v3.0

            # 训练！

                junweil@precognition-gpu3:~/projects/wbc_manipulation/Isaac-GR00T$ uv run torchrun --nproc_per_node=2 gr00t/experiment/launch_finetune.py      --base-model-path ../GR00T-N1.6-3B      --dataset-path ../junweiliang/wbc_close_washer_door      --embodiment-tag NEW_EMBODIMENT      --modality-config-path my_configs/g1_dex3_gripper_homie.py      --save-total-limit 3      --learning_rate 1e-4      --save-steps 2000      --max-steps 10000      --use-wandb      --warmup_ratio 0.05      --weight_decay 1e-5      --global-batch-size 32      --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08      --dataloader-num-workers 6  --output-dir experiments/mygpu3_wbc_close_washer_door

                # 2xA6000, b=32,s=10k, pick_up_object_from_ground也要3小时，和2x4090差不多，但是效率比较高，270w/300w

                # open loop eval, train/val

                    junweil@precognition-gpu3:~/projects/wbc_manipulation/Isaac-GR00T$ uv run python gr00t/eval/open_loop_eval.py     --dataset-path ~/projects/wbc_manipulation/junweiliang/wbc_close_washer_door_val0.1  --embodiment-tag NEW_EMBODIMENT     --model-path experiments/mygpu3_wbc_close_washer_door/checkpoint-10000/     --traj-ids 0 1 2 3 4 5     --action-horizon 16 --save-plot-path experiments/mygpu3_wbc_close_washer_door/open_loop_val_ep0-5.jpg


            # 8卡, batch_size=128, (256/512 OOM), s=10k
                bs=64,2xGPU, ok
                bs=128,2xGPU, OOM, bs=128,8xGPU, OOM
                bs=256, 8xGPU, gradient-accumulation-steps 4, export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OOM
                bs=128, 2xGPU, --gradient-accumulation-steps 8 OOM??

                # bs=64, 2卡，训练5task, dataworker=6, 10k step, (worker=4会经常要等shard)
                    #  CPU利用率44%, 内存使用40GB

                junweil@precognition-gpu3:~/projects/wbc_manipulation/Isaac-GR00T$ uv run torchrun --nproc_per_node=2 gr00t/experiment/launch_finetune.py      --base-model-path ../GR00T-N1.6-3B      --dataset-path ../junweiliang/wbc_5tasks      --embodiment-tag NEW_EMBODIMENT      --modality-config-path my_configs/g1_dex3_gripper_homie.py      --save-total-limit 3      --learning_rate 1e-4      --save-steps 2000      --max-steps 10000      --use-wandb      --warmup_ratio 0.05      --weight_decay 1e-5      --global-batch-size 64 --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08      --dataloader-num-workers 6  --output-dir experiments/mygpu3_wbc_5tasks_bs64_s10k

                # 训练5 tasks, open loop 测试 0-5, 100-5, 200-5 的episode
                    # validation 测试 0 1 8 9 12 13 19 20 24 25

                        junweil@precognition-gpu3:~/projects/wbc_manipulation/Isaac-GR00T$ uv run python gr00t/eval/open_loop_eval.py     --dataset-path ~/projects/wbc_manipulation/junweiliang/wbc_5tasks_val0.1  --embodiment-tag NEW_EMBODIMENT     --model-path experiments/mygpu3_wbc_5tasks_bs64_s10k/checkpoint-10000/     --traj-ids 0 1 8 9 12 13 19 20 24 25     --action-horizon 16 --save-plot-path experiments/mygpu3_wbc_5tasks_bs64_s10k/open_loop_val_5tasks.jpg

                    # 查看validation episode数量:

                        (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/inspect_lerobot_dataset.py --repo-id junweiliang/wbc_5tasks_val0.1

                        [Episodes per Task]
                        - 'Unknown Task (Index close_washer_door)': 4 episodes
                        - 'Unknown Task (Index move_and_open_pot)': 7 episodes
                        - 'Unknown Task (Index move_box)': 8 episodes
                        - 'Unknown Task (Index open_washer_door)': 3 episodes
                        - 'Unknown Task (Index pick_up_object_from_ground)': 6 episodes

                    # 图像：
                        junweiliang@work_laptop:~/Downloads$ scp -r junweil@gpu3.precognition.team:~/projects/wbc_manipulation/Isaac-GR00T/experiments/mygpu3_wbc_5tasks_bs64_s10k/*.jpg .

            # [04/17/2026] 观察：单卡batch_size 64可以跑，但是8卡128 都OOM，肯定有bug, gradient accumulation也不work。后续用N1.7的code

            # gpu3 单卡跑：
                junweil@precognition-gpu3:~/projects/wbc_manipulation/Isaac-GR00T$ CUDA_VISIBLE_DEVICES=2 uv run torchrun --nproc_per_node=1 --master_port=29501 gr00t/experiment/launch_finetune.py      --base-model-path ../GR00T-N1.6-3B      --dataset-path ../junweiliang/wbc_move_and_open_pot      --embodiment-tag NEW_EMBODIMENT      --modality-config-path my_configs/g1_dex3_gripper_homie.py      --save-total-limit 3      --learning_rate 1e-4      --save-steps 2000      --max-steps 10000      --use-wandb      --warmup_ratio 0.05      --weight_decay 1e-5      --global-batch-size 64      --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08      --dataloader-num-workers 6  --output-dir experiments/mygpu3_wbc_move_and_open_pot_bs64_s10k
    ```
+ 以下数据处理需要用的代码repo: `https://github.com/JunweiLiang/humanoid_teleop`
+ 可视化我们中期采集的数据
```
    # 5个任务的可视化视频:
        https://drive.google.com/drive/folders/120JGNOUmESJtJZ3OTWuyyHOllV9xOLBc
    # 可视化数据例子，move_box

        (tv) junweil@office-precognition:~/projects/humanoid_teleop$ python g1_realrobot/visualize_wbc_episodes.py ~/projects/huawei_data/wbc_task5/move_box/episode_0015/data.json assets/g1/g1_body29_hand14.urdf --fps 60 --image_path ~/projects/huawei_data/wbc_task5/move_box/episode_0015/colors/ --hand_type dex3 --show_states

    # Note
        # 底层用homie locomotion
        # 视频数据用realsense d435 640x480，视角小，数据采集是60fps; 尤其是move_box， states 看起来更准, action不对
        # 每一帧数据包括 (states 下面)
            # left_arm/right_arm -> qpos [7维度] shoulder pyr, elbow, wrist pyr
            # left_ee/right_ee -> qpos [7维度]多个手指
            # waist -> qpos [3维度, yaw, roll, pitch]，应该只有yaw非零
            # leg -> qpos [12维度]， 双腿+脚2
            # actions 下面
                # left_trigger/right_trigger: 0-1
                # loco_cmd: [v_x, v_y, v_yaw, height 1.65 - 0.8]
```
+ 转换中期数据到LeRobot v2 格式
```
    # LeRoBot v2数据格式

    # 参考代码
        # 宇树官方遥操作采的数据转换代码
            # https://github.com/unitreerobotics/unitree_lerobot/tree/main?tab=readme-ov-file#23-%EF%B8%8F-data-conversion

        # Psi也有转换
            https://github.com/physical-superintelligence-lab/Psi0/blob/main/scripts/data/raw_to_lerobot_v2.py

    # 转换华为wbc 5 tasks数据

        (base) junweil@office-precognition:~/projects/huawei_data$ cp -r wbc_task5 wbc_task5_lerobotv2

        # 原来的数据集中可能有缺失的episode，要重新按顺序命名

        (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/sort_and_rename_folders.py --data_dir wbc_task5_lerobotv2/move_box/
            close_washer_door/          move_box/                   pick_up_object_from_ground/
            move_and_open_pot/          open_washer_door/

        # convert based on the data.json

            # install the lerobot package (copied from https://github.com/unitreerobotics/unitree_lerobot/unitree_lerobot/lerobot)

            (tv) junweil@office-precognition:~/projects/humanoid_teleop/g1_realrobot/lerobot$ pip install -e .

        # convert to LeRobot v2 and Gr00T complient (就是多一个modality.json)
            # https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md

        (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/convert_unitree_json_to_lerobot.py --raw-dir wbc_task5_lerobotv2/ --repo-id junweiliang/wbc_5tasks --downsample-factor 2 --use-future-state-as-action --valp 0.1 --repo-id-val junweiliang/wbc_5tasks_val0.1

            # LeRobot 会提取jpg 生成mp4文件

            # 要一个小时，数据会存在
                ~/.cache/huggingface/lerobot/junweiliang/wbc_5tasks

                [WARNING] Skipping Episode 168 (wbc_task5_lerobotv2/open_washer_door/episode_0000)
                  Reason: Shape mismatch. State: 29 (expected 43).
                  (This usually means hand tracking data was absent during recording).

                # 分一些到validation里
                  ~/.cache/huggingface/lerobot/junweiliang/wbc_5tasks_val0.1

                # 有一些episode可能手的states 没有录制，没有数据就跳过。lerobot会跳过这个episode

                # 把原始数据的state和action，字段复制补齐，这样两边都是49

                    # raw_state is 43D: Arms(14) + Hands(14) + Waist(3) + Legs(12)
                    # raw_action is 37D: Arms(14) + Hands(14) + Waist(3) + Triggers(2) + Loco(4)

                # 会按照 Gr00T说的，额外生成modality.json，还有旧版的.jsonl meta文件
                    # https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md


            # 转换完后查看数据集, 原本是5个任务一共281 episode，
                # train set 252
                    (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/inspect_lerobot_dataset.py --repo-id junweiliang/wbc_5tasks

                    [Overall Stats]
                    - Total Episodes : 252
                    - Total Frames   : 124560
                    - FPS            : 30

                    [Available Tasks]
                    - Index 0: 'close_washer_door'
                    - Index 1: 'move_and_open_pot'
                    - Index 2: 'move_box'
                    - Index 3: 'open_washer_door'
                    - Index 4: 'pick_up_object_from_ground'

                    [Episodes per Task]
                    - 'close_washer_door': 50 episodes
                    - 'move_and_open_pot': 50 episodes
                    - 'move_box': 49 episodes
                    - 'open_washer_door': 48 episodes
                    - 'pick_up_object_from_ground': 55 episodes

                    [Episode Length Stats (Frames)]
                    - Average: 494.3 frames
                    - Min    : 143 frames
                    - Max    : 1381 frames

                # val set 28 episode

                    (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/inspect_lerobot_dataset.py --repo-id junweiliang/wbc_5tasks_val0.1

                    [Episodes per Task]
                    - 'close_washer_door': 4 episodes
                    - 'move_and_open_pot': 7 episodes
                    - 'move_box': 8 episodes
                    - 'open_washer_door': 3 episodes
                    - 'pick_up_object_from_ground': 6 episodes

            # 可视化lerobot 数据

                (tv) junweil@office-precognition:~/projects$ python ~/projects/humanoid_teleop/g1_realrobot/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py --repo-id junweiliang/wbc_5tasks --episode-index 0

                    # 会打开rerun窗口，看到视频，还有各个关节的曲线图


            # gr00T 可能会把全部5个任务一起训练。我们生成数据集的时候挑单个任务,比如关闭洗衣机门，搬箱子，捡起物体

                (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/convert_unitree_json_to_lerobot.py --raw-dir wbc_task5_lerobotv2/ --repo-id junweiliang/wbc_close_washer_door --downsample-factor 2 --use-future-state-as-action --valp 0.1 --repo-id-val junweiliang/wbc_close_washer_door_val0.1 --tasks close_washer_door

                (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/convert_unitree_json_to_lerobot.py --raw-dir wbc_task5_lerobotv2/ --repo-id junweiliang/wbc_move_box --downsample-factor 2 --use-future-state-as-action --valp 0.1 --repo-id-val junweiliang/wbc_move_box_val0.1 --tasks move_box

                (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/convert_unitree_json_to_lerobot.py --raw-dir wbc_task5_lerobotv2/ --repo-id junweiliang/wbc_pick_up_object_from_ground --downsample-factor 2 --use-future-state-as-action --valp 0.1 --repo-id-val junweiliang/wbc_pick_up_object_from_ground_val0.1 --tasks pick_up_object_from_ground

                # 再可视化一下
                    (tv) junweil@office-precognition:~/projects$ python ~/projects/humanoid_teleop/g1_realrobot/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py --repo-id junweiliang/wbc_pick_up_object_from_ground --episode-index 3

                # 格式检查
                    (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/inspect_lerobot_dataset.py --repo-id junweiliang/wbc_pick_up_object_from_ground



            # 上述代码得到的数据是LeRobot v3，需要转回v2
                # v3 和v2区别： https://io-ai.tech/platform/guides/Pipeline/LeRobot/LeRobotV2V3Format/
                    # 主要是视频，v2按照episode存， v3弄成大文件
                (tv) junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T$ python scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id junweiliang/wbc_pick_up_object_from_ground

                # 会直接修改原有数据集：/home/junweil/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground

                # 生成v3.0备份：/home/junweil/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground_v3.0/

                # 还缺modality.json

                (tv) junweil@office-precognition:~/.cache/huggingface/lerobot/junweiliang$ cp wbc_move_and_open_pot_v3.0/meta/modality.json wbc_move_and_open_pot/meta/


                # validation
                    (tv) junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T$ python scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id junweiliang/wbc_pick_up_object_from_ground_val0.1

                    (tv) junweil@office-precognition:~/.cache/huggingface/lerobot/junweiliang$ cp wbc_pick_up_object_from_ground_val0.1_v3.0/meta/modality.json wbc_pick_up_object_from_ground_val0.1/meta/

            (tv) junweil@office-precognition:~/.cache/huggingface/lerobot/junweiliang$ ls
                # 一共24， 5个任务合集+分开，每个有3.0/2.1的train+val
                wbc_5tasks                         wbc_move_and_open_pot              wbc_open_washer_door
                wbc_5tasks_v3.0                    wbc_move_and_open_pot_v3.0         wbc_open_washer_door_v3.0
                wbc_5tasks_val0.1                  wbc_move_and_open_pot_val0.1       wbc_open_washer_door_val0.1
                wbc_5tasks_val0.1_v3.0             wbc_move_and_open_pot_val0.1_v3.0  wbc_open_washer_door_val0.1_v3.0
                wbc_close_washer_door              wbc_move_box                       wbc_pick_up_object_from_ground
                wbc_close_washer_door_v3.0         wbc_move_box_v3.0                  wbc_pick_up_object_from_ground_v3.0
                wbc_close_washer_door_val0.1       wbc_move_box_val0.1                wbc_pick_up_object_from_ground_val0.1
                wbc_close_washer_door_val0.1_v3.0  wbc_move_box_val0.1_v3.0           wbc_pick_up_object_from_ground_val0.1_v3.0

            # pack the data, 把全部任务、单个任务都一并打包
                (tv) junweil@office-precognition:~/.cache/huggingface/lerobot$ tar -zcvf lerobot_wbc_datasets_v3+v2_5tasks+5singletask.tgz junweiliang/

```
