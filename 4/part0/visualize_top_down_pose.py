import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os

def create_top_down_pose(tag_size=0.1, camera_height=0.5):
    """
    创建俯视ArUco标签时的相机姿态
    
    Args:
        tag_size: ArUco标签的大小(边长)
        camera_height: 相机距离桌面的高度
    
    Returns:
        c2w: 相机到世界坐标的变换矩阵 (4x4)
    """
    # 相机位置 - 在标签正上方
    camera_position = np.array([0, 0, camera_height])
    
    # 相机朝向 - 朝下看标签中心
    # 在相机坐标系中:
    # Z轴: 向前(朝向标签)
    # Y轴: 向上
    # X轴: 向右
    
    # 由于相机在标签正上方朝下看:
    # 相机的Z轴(光轴)指向负Z方向(向下)
    # 相机的Y轴指向世界坐标的-Y方向(向前)
    # 相机的X轴指向世界坐标的+X方向(向右)
    
    # 旋转矩阵: 世界坐标到相机坐标
    # 相机朝下看，绕X轴旋转180度
    c2w = np.array([
        [1, 0, 0, 0],          # X轴方向 - 向右
        [0, -1, 0, 0],         # Y轴方向 - 向前(世界坐标的-Y)
        [0, 0, -1, camera_height],  # Z轴方向 - 向下(世界坐标的-Z)
        [0, 0, 0, 1]
    ])
    
    return c2w

def visualize_camera_pose():
    """
    可视化俯视ArUco标签时的相机姿态
    """
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    ax.set_zlim([0, 0.6])
    
    # 设置坐标轴标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Top-down View of ArUco Tag and Camera Pose')
    
    # 绘制桌面 (XY平面)
    x = np.linspace(-0.3, 0.3, 2)
    y = np.linspace(-0.3, 0.3, 2)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.2, color='gray')
    
    # 绘制ArUco标签 (在XY平面上)
    tag_size = 0.1
    half_size = tag_size / 2
    tag_corners = np.array([
        [-half_size, -half_size, 0],  # 左下角
        [half_size, -half_size, 0],   # 右下角
        [half_size, half_size, 0],    # 右上角
        [-half_size, half_size, 0],   # 左上角
        [-half_size, -half_size, 0]   # 回到起点闭合
    ])
    ax.plot(tag_corners[:, 0], tag_corners[:, 1], tag_corners[:, 2], 'r-', linewidth=2, label='ArUco Tag')
    
    # 标记标签的坐标轴
    # 标签坐标系原点在中心
    ax.quiver(0, 0, 0, 0.05, 0, 0, color='red', arrow_length_ratio=0.1)    # X轴
    ax.quiver(0, 0, 0, 0, 0.05, 0, color='green', arrow_length_ratio=0.1)  # Y轴
    ax.quiver(0, 0, 0, 0, 0, 0.05, color='blue', arrow_length_ratio=0.1)   # Z轴
    ax.text(0.06, 0, 0, 'X_tag', color='red')
    ax.text(0, 0.06, 0, 'Y_tag', color='green')
    ax.text(0, 0, 0.06, 'Z_tag', color='blue')
    
    # 相机位置
    camera_height = 0.5
    camera_pos = np.array([0, 0, camera_height])
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], color='blue', s=100, label='Camera')
    
    # 获取相机姿态矩阵
    c2w = create_top_down_pose(tag_size, camera_height)
    
    # 绘制相机坐标轴
    # 相机坐标系原点
    cam_origin = c2w[:3, 3]
    
    # 相机坐标轴方向 (从c2w矩阵的前三列获取)
    cam_x = c2w[:3, 0] * 0.1  # X轴方向，缩放0.1倍
    cam_y = c2w[:3, 1] * 0.1  # Y轴方向，缩放0.1倍
    cam_z = c2w[:3, 2] * 0.1  # Z轴方向，缩放0.1倍
    
    ax.quiver(cam_origin[0], cam_origin[1], cam_origin[2], 
              cam_x[0], cam_x[1], cam_x[2], 
              color='orange', arrow_length_ratio=0.2)
    ax.quiver(cam_origin[0], cam_origin[1], cam_origin[2], 
              cam_y[0], cam_y[1], cam_y[2], 
              color='cyan', arrow_length_ratio=0.2)
    ax.quiver(cam_origin[0], cam_origin[1], cam_origin[2], 
              cam_z[0], cam_z[1], cam_z[2], 
              color='purple', arrow_length_ratio=0.2)
    
    ax.text(cam_origin[0] + cam_x[0], cam_origin[1] + cam_x[1], cam_origin[2] + cam_x[2], 
            'X_cam', color='orange')
    ax.text(cam_origin[0] + cam_y[0], cam_origin[1] + cam_y[1], cam_origin[2] + cam_y[2], 
            'Y_cam', color='cyan')
    ax.text(cam_origin[0] + cam_z[0], cam_origin[1] + cam_z[1], cam_origin[2] + cam_z[2], 
            'Z_cam', color='purple')
    
    # 绘制相机到标签中心的视线
    ax.plot([camera_pos[0], 0], [camera_pos[1], 0], [camera_pos[2], 0], 
            'b--', alpha=0.7, label='Line of sight')
    
    # 添加图例
    ax.legend()
    
    # 保存图像
    plt.savefig('top_down_camera_pose.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to 'top_down_camera_pose.png'")
    
    # 显示相机姿态矩阵
    print("\nCamera-to-World Transformation Matrix (c2w):")
    print(c2w)
    
    # 解释矩阵含义
    print("\nExplanation:")
    print("- The camera is positioned directly above the ArUco tag at (0, 0, {})".format(camera_height))
    print("- The camera is looking straight down at the tag")
    print("- Camera X-axis (red arrow): Points to the right (same as world X)")
    print("- Camera Y-axis (cyan arrow): Points forward in camera view, which is -Y in world coordinates")
    print("- Camera Z-axis (purple arrow): Points downward, which is -Z in world coordinates")
    print("- This corresponds to a camera directly above the tag, pointing down at it")
    
    plt.show()

if __name__ == "__main__":
    visualize_camera_pose()