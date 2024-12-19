mkdir -p ./p_weight
cd ./p_weight

for i in {0..31}; do
  wget "https://huggingface.co/Zkli/init_mod_test/resolve/main/Layer${i}_up_P_matrix.pt"
  wget "https://huggingface.co/Zkli/init_mod_test/resolve/main/Layer${i}_gate_P_matrix.pt"
  wget "https://huggingface.co/Zkli/init_mod_test/resolve/main/Layer${i}_self_P_matrix.pt"
done
