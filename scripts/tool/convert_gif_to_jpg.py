import os
from PIL import Image
import shutil

def convert_gif_to_jpg(directory):
    """
    将目录中所有实际为GIF格式但扩展名为.jpg的文件转换为真正的JPG格式
    """
    gif_files = []
    
    # 遍历目录中的所有.jpg文件
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            filepath = os.path.join(directory, filename)
            
            # 检查文件头是否为GIF格式
            try:
                with open(filepath, 'rb') as f:
                    header = f.read(3)
                    if header == b'GIF':
                        gif_files.append(filepath)
            except Exception as e:
                print(f"读取文件 {filepath} 时出错: {e}")
    
    print(f"找到 {len(gif_files)} 个GIF格式的.jpg文件")
    
    # 转换每个GIF文件为JPG
    converted_count = 0
    for filepath in gif_files:
        try:
            # 打开GIF图像并转换为RGB模式
            img = Image.open(filepath)
            
            # 如果是动画GIF，只取第一帧
            if hasattr(img, 'is_animated') and img.is_animated:
                img.seek(0)  # 跳转到第一帧
            
            # 转换为RGB模式（JPG不支持透明度）
            if img.mode in ('RGBA', 'LA', 'P'):
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 保存为JPG格式
            img.save(filepath, 'JPEG')
            converted_count += 1
            print(f"已转换: {os.path.basename(filepath)}")
            
        except Exception as e:
            print(f"转换文件 {filepath} 时出错: {e}")
    
    print(f"成功转换 {converted_count} 个文件")
    return converted_count

if __name__ == "__main__":
    train_dir = "datasets/images/train"
    val_dir = "datasets/images/val"
    
    print("处理训练集...")
    train_converted = convert_gif_to_jpg(train_dir)
    
    print("\n处理验证集...")
    val_converted = convert_gif_to_jpg(val_dir)
    
    print(f"\n总计转换: {train_converted + val_converted} 个文件")