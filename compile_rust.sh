maturin build -m ./vsp_custom_env/Cargo.toml --release
pip install vsp_custom_env/target/wheels/vsp_env-0.1.1-cp36-cp36m-manylinux_2_27_x86_64.whl --force-reinstall
#python ./src/vsp/train_model.py



