"""
Windows 系统获取 USB 相机数据 - 使用 OpenCV
需要安装: pip install opencv-python
"""
import cv2

def capture_camera(camera_id=1):
    # 打开相机（0 表示第一个相机，如果有多个相机可以尝试 1, 2, ...）
    cap = cv2.VideoCapture(camera_id)
    
    # 检查相机是否成功打开
    if not cap.isOpened():
        print("错误：无法打开相机")
        return
    
    # 设置相机参数（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)   # 设置高度
    cap.set(cv2.CAP_PROP_FPS, 30)             # 设置帧率
    
    print(f"相机 {camera_id} 已打开")
    print("按 'q' 键退出，按 's' 键保存当前帧")
    
    frame_count = 0
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        if not ret:
            print("错误：无法读取帧")
            break
        
        # 显示图像
        cv2.imshow('USB Camera', frame)
        
        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):  # 按 's' 保存当前帧
            filename = f'capture_{frame_count:04d}.jpg'
            cv2.imwrite(filename, frame)
            print(f"已保存: {filename}")
            frame_count += 1
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def capture_and_save(camera_id=1):
    """捕获并保存图像（与capture_camera功能相同）"""
    capture_camera(camera_id)

def list_cameras():
    """列出所有可用的相机"""
    available_cameras = []
    
    # 尝试打开前 10 个相机索引
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    print(f"找到 {len(available_cameras)} 个相机:")
    for cam_id in available_cameras:
        print(f"  相机索引: {cam_id}")
    
    return available_cameras

if __name__ == "__main__":
    # 列出所有相机
    list_cameras()
    
    # 捕获相机数据（使用相机1）
    capture_camera(camera_id=1)
    
    # 或者捕获并保存
    # capture_and_save()

