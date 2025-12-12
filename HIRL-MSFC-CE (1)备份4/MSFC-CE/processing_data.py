import scipy.io as sio
import numpy as np
import os


def process_and_save_selected():
    # 1. 确保输出目录存在
    output_dir = './plot_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 需要处理的文件名列表
    files = [
        'bandwidth_resource_comp', 'bandwidth_load_var',
        'cpu_load_var', 'cpu_resource_comp',
        'memory_load_var', 'memory_resource_comp'
    ]

    for name in files:
        try:
            # 2. 加载数据
            # 注意：这里假设源文件在 ./output/ 目录下
            src_path = f'./output/{name}.mat'
            if not os.path.exists(src_path):
                print(f"Warning: File not found {src_path}")
                continue

            data = sio.loadmat(src_path)[name]

            # 3. 数据处理与降采样
            # 先 flatten 变成一维数组，防止因为是 (1, N) 还是 (N, 1) 导致切片方向错误
            # MATLAB: 50:50:400 -> Python Index: 49::50
            flat_data = data.flatten()

            # 检查数据长度是否足够
            if len(flat_data) < 50:
                print(f"Warning: Data {name} is too short to sample.")
                selected_values = np.array([])
            else:
                selected_values = flat_data[49::50]

            # 4. 保存数据
            dst_path = f'{output_dir}/{name}_selectedValues.mat'
            sio.savemat(dst_path, {f'{name}_selectedValues': selected_values})
            print(f"Processed and saved: {dst_path} (Shape: {selected_values.shape})")

        except Exception as e:
            print(f"Error processing {name}: {e}")


if __name__ == "__main__":
    process_and_save_selected()