# MiniMind Web 前端 + 后端一体化镜像
# 构建方式：由 run_all_npu.sh stage_web 自动执行
#   1. npm run build → 生成 dist/
#   2. 复制 serve_train_manager.py 到构建上下文
#   3. docker build
FROM m.daocloud.io/docker.io/library/nginx:alpine

# Python + 后端依赖
RUN apk add --no-cache python3 py3-pip \
    && pip3 install --no-cache-dir --break-system-packages \
       fastapi uvicorn pydantic

# 前端静态文件
COPY dist/ /usr/share/nginx/html/

# Nginx 配置模板（envsubst 替换 ${VLLM_UPSTREAM}）
COPY nginx.conf /etc/nginx/templates/default.conf.template

# 后端
COPY serve_train_manager.py /app/scripts/serve_train_manager.py

# 入口脚本
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# VLLM_UPSTREAM: docker-compose 中设为 vllm:8000，单独运行时默认 127.0.0.1:8000
ENV VLLM_UPSTREAM=127.0.0.1:8000
EXPOSE 80
ENTRYPOINT ["/entrypoint.sh"]
