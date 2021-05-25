
OpenMVS 注释版

建议：

1、在Ubuntu16.04版本下安装，依赖库在这个版本下更适合。其他版本自己可以尝试。

2、第三步main_path='pwd' 这个其实是设置的是vcglib所在的路径，可以在VCGLib安装之后再指定， 例如我的vcg在/home/code/mvs下，main_path='/home/code/mvs'

3、boost 安装过程中如果有问题可以尝试不同版本，比如升级到1.6.3 。

4、install时候如果opencv报一些未定义错误，需要自己换更高级别的版本，比如opencv3或4从源码安装。

附：

**课程目录**：

<img src="https://github.com/ReeseL/pictures_lib/raw/main/03.png" alt="c1_22" style="zoom:200%;" />

**课程入口**：

链接：https://appafc4omci9700.h5.xiaoeknow.com

扫码加入：

![in](https://github.com/ReeseL/pictures_lib/raw/main/04.jpg)

---

# OpenMVS: open Multi-View Stereo reconstruction library

## Introduction

[OpenMVS (Multi-View Stereo)](http://cdcseacave.github.io/openMVS) is a library for computer-vision scientists and especially targeted to the Multi-View Stereo reconstruction community. While there are mature and complete open-source projects targeting Structure-from-Motion pipelines (like [OpenMVG](https://github.com/openMVG/openMVG)) which recover camera poses and a sparse 3D point-cloud from an input set of images, there are none addressing the last part of the photogrammetry chain-flow. *OpenMVS* aims at filling that gap by providing a complete set of algorithms to recover the full surface of the scene to be reconstructed. The input is a set of camera poses plus the sparse point-cloud and the output is a textured mesh. The main topics covered by this project are:

- **dense point-cloud reconstruction** for obtaining a complete and accurate as possible point-cloud
- **mesh reconstruction** for estimating a mesh surface that explains the best the input point-cloud
- **mesh refinement** for recovering all fine details
- **mesh texturing** for computing a sharp and accurate texture to color the mesh

See the complete [documentation](https://github.com/cdcseacave/openMVS/wiki) on wiki.

## Build

See the [building](https://github.com/cdcseacave/openMVS/wiki/Building) wiki page. Windows and Ubuntu x64 continuous integration status [![Build Status](https://ci.appveyor.com/api/projects/status/github/cdcseacave/openmvs?branch=master&svg=true)](https://ci.appveyor.com/project/cdcseacave/openmvs)
Automatic Windows x64 binary builds can be found for each commit on its Appveyor Artifacts page.

## Example

See the usage [example](https://github.com/cdcseacave/openMVS/wiki/Usage) wiki page.

## License

See the [copyright](https://github.com/cdcseacave/openMVS/blob/master/COPYRIGHT.md) file.

## Contact

openmvs[AT]googlegroups.com
