![Cover Art](https://alain.xyz/blog/raw-vulkan/assets/cover.jpg)

# Vulkan Seed

[![cmake-img]][cmake-url]
[![License][license-img]][license-url]

A Vulkan repo you can use to get started with your own renderer.

## Setup

First install:

- [Git](https://git-scm.com/)

- [CMake](https://cmake.org)

- An IDE such as [Visual Studio](https://visualstudio.microsoft.com/downloads/), [XCode](https://developer.apple.com/xcode/), or a compiler such as [GCC](https://gcc.gnu.org/).

Then type the following in your [terminal](https://hyper.is/).

```bash
# 🐑 Clone the repo
git clone https://github.com/alaingalvan/vulkan-seed --recurse-submodules

# 💿 go inside the folder
cd vulkan-seed

# 👯 If you forget to `recurse-submodules` you can always run:
git submodule update --init

# 👷 Make a build folder
mkdir build
cd build

# 🖼️ To build your Visual Studio solution on Windows x64
cmake .. -A x64

# 🍎 To build your XCode project on Mac OS
cmake .. -G Xcode

# 🐧 To build your .make file on Linux
cmake ..

# 🔨 Build on any platform:
cmake --build .
```

> Refer to [this blog post on designing C++ libraries and apps](https://alain.xyz/blog/designing-a-cpp-library) for more details on CMake, Git Submodules, etc.

## Project Layout

As your project becomes more complex, you'll want to separate files and organize your application to something more akin to a game or renderer, check out this post on [game engine architecture](https://alain.xyz/blog/game-engine-architecture) and this one on [real time renderer architecture](https://alain.xyz/blog/realtime-renderer-architectures) for more details.

```bash
├─ 📂 external/                    # 👶 Dependencies
│  ├─ 📁 crosswindow/                    # 🖼️ OS Windows
│  ├─ 📁 crosswindow-graphics/           # 🎨 Vulkan Surface Creation
│  └─ 📁 glm/                            # ➕ Linear Algebra
├─ 📂 src/                         # 🌟 Source Files
│  ├─ 📄 Utils.h                         # ⚙️ Utilities (Load Files, Check Shaders, etc.)
│  ├─ 📄 Renderer.h                      # 🔺 Triangle Draw Code
│  ├─ 📄 Renderer.cpp                    # -
│  └─ 📄 Main.cpp                        # 🏁 Application Main
├─ 📄 .gitignore                   # 👁️ Ignore certain files in git repo
├─ 📄 CMakeLists.txt               # 🔨 Build Script
├─ 📄 license.md                   # ⚖️ Your License (Unlicense)
└─ 📃readme.md                     # 📖 Read Me!
-->

[cmake-img]: https://img.shields.io/badge/cmake-3.6-1f9948.svg?style=flat-square
[cmake-url]: https://cmake.org/
[license-img]: https://img.shields.io/:license-mit-blue.svg?style=flat-square
[license-url]: https://opensource.org/licenses/MIT
