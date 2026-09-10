# Python 3.14 is the interpreter the committed artefacts were generated with.
# requirements.txt, requirements-dev.txt and .github/workflows/ci.yml name the
# same version; tests/test_python_pin.py fails if the four ever drift apart.
FROM python:3.14-slim

WORKDIR /app

# Dev tooling is installed too, so the image can run the test suite and
# tools/verify_artifacts.py rather than only the pipeline.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY . .

RUN chmod +x run_all_experiments.sh

CMD ["bash", "run_all_experiments.sh"]
