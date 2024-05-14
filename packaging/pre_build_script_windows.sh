python -m pip install -U numpy packaging pyyaml setuptools wheel

# Install TRT 10 from PyPi
python -m pip install tensorrt==10.0.0b6 tensorrt-${CU_VERSION::4}-bindings==10.0.0b6 tensorrt-${CU_VERSION::4}-libs==10.0.0b6 --extra-index-url https://pypi.nvidia.com

choco install bazelisk -y

cat toolchains/ci_workspaces/WORKSPACE.win.release.tmpl | envsubst > WORKSPACE

echo "RELEASE=1" >> ${GITHUB_ENV}
