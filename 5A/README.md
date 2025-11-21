# CS 180 Project 5A - Diffusion Models Showcase

## 项目概述 (Project Overview)

这是UC Berkeley CS180课程 Project 5A 的完整实现展示。项目展示了从Part 1.1到Part 1.7的所有结果。

This is a comprehensive showcase of CS180 Project 5A implementation, demonstrating Parts 1.1 through 1.7.

## 📊 项目包含内容 (Project Contents)

### Part 1.1 - Forward Process (加噪过程)
- 实现了前向扩散过程
- 展示了Campanile在不同噪声水平 (t=250, 500, 750) 的效果

### Part 1.2 - Classical Denoising (经典去噪)
- 使用高斯模糊进行图像去噪
- 对比了不同噪声水平下的结果
- 说明了为什么深度学习方法更优

### Part 1.3 - One-Step Denoising (单步去噪)
- 使用预训练的UNet进行单步去噪
- 对比Gaussian blur和UNet的效果
- 证明了深度学习在去噪中的优势

### Part 1.4 - Iterative Denoising (迭代去噪)
- 完整实现DDPM采样循环
- 展示了逐步去噪的过程
- 对比了四种去噪方法：
  - 原始噪声图像
  - 高斯模糊
  - 单步UNet去噪
  - 迭代DDPM (最优效果)

### Part 1.5 - Unconditional Generation (无条件生成)
- 从纯噪声生成5个样本
- 使用"a high quality photo"作为弱条件

### Part 1.6 - Classifier-Free Guidance (CFG) (分类器自由引导)
- 实现CFG采样方法，指导强度 γ=7
- 展示了5个高质量生成样本
- 对比了有无CFG的效果差异

### Part 1.7 - Image-to-Image Translation (图像到图像翻译)
- 实现SDEdit算法
- 展示了Campanile在不同编辑程度下的变化
- 包含其他两个测试图像的编辑结果
- 演示了噪声级别对编辑强度的精细控制

## 🚀 如何查看展示网页 (How to View)

### 方法1: 直接打开HTML文件
双击或在浏览器中打开 `index.html` 文件

### 方法2: 使用Python本地服务器
```bash
cd d:/Code/cs180/ezoiLZH.github.io/5A
python server.py
```
然后在浏览器中打开 `http://localhost:8000`

### 方法3: 使用VS Code Live Server
如果安装了Live Server扩展，右键点击index.html并选择"Open with Live Server"

## 📁 文件结构 (File Structure)

```
5A/
├── index.html                          # 主展示网页
├── server.py                           # 本地web服务器脚本
├── README.md                           # 本文件
├── cs180_proj5a_release.ipynb         # 原始Jupyter notebook
├── campanile.jpg                       # 测试图像
│
├── [Part 1.1] Forward Process Images
│   ├── noise_level_250.png
│   ├── noise_level_500.png
│   └── noise_level_750.png
│
├── [Part 1.2] Classical Denoising Images
│   ├── denoised_noise_level_250.png
│   ├── denoised_noise_level_500.png
│   └── denoised_noise_level_750.png
│
├── [Part 1.3] One-Step Denoising Images
│   ├── denoised_unet_level_250.png
│   ├── denoised_unet_level_500.png
│   └── denoised_unet_level_750.png
│
├── [Part 1.4] Iterative Denoising Images
│   ├── iterative_denoised_image_step_0.png
│   ├── iterative_denoised_image_step_5.png
│   ├── iterative_denoised_image_step_10.png
│   ├── iterative_denoised_image_step_15.png
│   ├── iterative_denoised_image_step_20.png
│   ├── iterative_denoised_image.png
│   ├── one_step_denoised_image.png
│   ├── blur_filtered_image.png
│   └── original_noisy_image.png
│
├── [Part 1.5] Unconditional Generation Images
│   ├── generated_image_sample_1.png
│   ├── generated_image_sample_2.png
│   ├── generated_image_sample_3.png
│   ├── generated_image_sample_4.png
│   └── generated_image_sample_5.png
│
├── [Part 1.6] CFG Generation Images
│   ├── cfg_generated_image_sample_1.png
│   ├── cfg_generated_image_sample_2.png
│   ├── cfg_generated_image_sample_3.png
│   ├── cfg_generated_image_sample_4.png
│   └── cfg_generated_image_sample_5.png
│
└── [Part 1.7] Image-to-Image Translation Images
    ├── edited_image_i_start_*.png (多个文件)
    └── ...
```

## 🎨 网页特性 (Website Features)

✅ **响应式设计** - 适配桌面、平板和手机屏幕
✅ **高质量图像展示** - 网格布局，光滑悬停效果
✅ **详细文字分析** - 每个部分都包含深入的技术分析
✅ **对比视图** - 并排展示不同方法的结果
✅ **现代UI设计** - 渐变背景，圆角卡片，阴影效果

## 📊 主要发现和洞察 (Key Findings)

### 1. 去噪效果对比
- **高斯模糊**: 在低噪声下有效，但在高噪声下完全失效
- **单步UNet**: 大幅改进但缺乏细节
- **迭代DDPM**: 几乎完美恢复原始图像

### 2. CFG的影响
- 无CFG: 图像模糊，缺乏细节
- CFG (γ=7): 清晰、高对比度、详细的细节

### 3. SDEdit控制
- 低噪声级别 (i_start=1-3): 保留原始结构
- 中等噪声 (i_start=5-10): 平衡的编辑效果
- 高噪声 (i_start=20): 创意转变

## 🔧 技术栈 (Tech Stack)

- **Model**: DeepFloyd IF (Stability AI)
- **Implementation**: PyTorch, Diffusers
- **Visualization**: HTML5, CSS3
- **Infrastructure**: Jupyter Notebook

## 📝 实现细节 (Implementation Details)

### 核心算法
- ✅ Forward diffusion process
- ✅ DDPM sampling loop with skipped timesteps
- ✅ Classifier-free guidance (CFG)
- ✅ SDEdit for image editing
- ✅ Iterative denoising with variance prediction

### 关键参数
- Random Seed: 100
- CFG Scale (γ): 7
- Denoising Steps: 33 (stride=30 from 990 to 0)
- Image Resolution: 64×64 (Stage 1)

## 🎓 学习资源 (Learning Resources)

关键论文和资源:
1. [DDPM: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
2. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
3. [SDEdit: Image Synthesis and Editing with Diffusion Models](https://sde-image-editing.github.io/)
4. [Understanding Diffusion Models](https://yang-song.net/blog/2021/score/)

## 📝 许可证 (License)

UC Berkeley CS180 - Educational Use Only

---

**创建时间** (Created): 2025年11月20日
**项目范围** (Scope): Parts 1.1 - 1.7
**模型** (Model): DeepFloyd IF by Stability AI
