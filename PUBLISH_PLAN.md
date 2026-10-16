# GitHub 发布步骤

目标：将当前插件发布到 `xLmiler/Anima-Contact-Regional-LoRA`，仅包含本目录内的项目文件。

1. 检查项目文件、忽略规则及 GitHub 认证，排除凭据、模型权重和缓存文件。
2. 使用现有 GitHub 凭据核对目标账号与仓库状态。
3. 创建同名私有 GitHub 仓库，通过 Git 数据接口在 `main` 分支提交项目文件。
4. 按文件路径、数量及 Git 对象哈希核对远程内容，交付仓库链接。

执行约束：当前环境禁止写入本地 `.git` 索引，因此使用 GitHub API 完成远程提交。
