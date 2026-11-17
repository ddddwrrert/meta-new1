# meta-new1

每道题单独从自己的 baseline 开始，相邻两档比较。这个仓库就是完整项目：克隆下来即可运行，不依赖别的目录。不要提交 `.venv` 和 `.env`。

需要 Python 3.10 或更新版本。Windows 和 Linux 都可以跑：Windows 用作业对象限制内存，Linux 用 `RLIMIT_AS`。

```text
meta_new/
  cli.py            命令入口
  config.py         配置
  models.py         候选、上下文、每题状态
  loop/             一阶循环和升降、ρ、ε
  meta/             ACT / HOLD / AVOID 和元准则
  generation/       Analyzer、Critic、Mutator、Verifier
  runtime/          沙箱和并行 evaluate
  problems/         TSP、背包和 baseline
configs/            运行配置
tests/              单测
```

```powershell
git clone https://github.com/ddddwrrert/meta-new1.git
cd meta-new1
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

Linux 或 macOS 把 `.venv\Scripts\python` 换成 `.venv/bin/python`。

单测不访问网络。要跑 `configs/demo.yaml`，把 `.env.example` 复制为仓库根目录的 `.env`，填入密钥，不要写进 yaml：

```text
LLM_API_KEY=你的密钥
```

然后：

```powershell
.venv\Scripts\python -m meta_new run --config configs\demo.yaml
```

没有这行密钥时，程序会停下来并提示设置 `LLM_API_KEY`。演示使用 DeepSeek，地址和模型已经写在 `configs/demo.yaml`。

ALE-Bench 十道题在 `tasks/ale_bench`。每题目录里有题面、`solution.cpp` 和 `evaluator/evaluator.py`。baseline 是这份 `solution.cpp`，一次评测打 50 个公开 seed，并带回每个 seed 的分数。替换只看这 50 个 seed 的平均分。`configs/ale_bench.yaml` 的 `max_budgets` 是 100：第 0 档跑 baseline，后面的档位改 C++。候选保持 C++20。

评测需要一台能导入 `ale_bench` 的 Linux Python，以及已经启动的 Docker。Windows 上使用 WSL 的 Ubuntu-24.04；默认的 `wsl` 发行版如果是 docker-desktop，跑不了这个环境。在那个 Linux 环境里安装 ALE-Bench，并准备镜像 `rust:1.79.0-buster` 和 `ale-bench:cpp20-202301`：

```sh
pip install "git+https://github.com/SakanaAI/ALE-Bench.git"
docker pull rust:1.79.0-buster
docker pull yimjk/ale-bench:cpp20-202301
docker tag yimjk/ale-bench:cpp20-202301 ale-bench:cpp20-202301
```

然后在本仓库的 `.env` 里写上这个解释器的绝对路径，例如：

```text
META_EVAL_PYTHON=/home/you/ale-bench/.venv/bin/python
```

Linux 本机可以直接把 `configs/ale_bench.yaml` 里的 `evaluator.via` 改成 `native`。题目数据在第一次评测时从 Hugging Face 下载。

```powershell
.venv\Scripts\python -m meta_new run --config configs\ale_bench.yaml
```
