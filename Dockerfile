FROM public.ecr.aws/lambda/python:3.11

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --only-binary :all: -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY configs/ ${LAMBDA_TASK_ROOT}/configs/

ENV MODEL_PACKAGE_GROUP="fraud-detection-baf"
ENV AWS_DEFAULT_REGION="us-east-1"
ENV LOSS_STRATEGY="scale_pos_weight"

CMD ["app.handler.handler"]
