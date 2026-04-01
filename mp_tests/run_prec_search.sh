cd mp_tests
for k in backprop hotspot particle_filter srad_v2; do
  echo "===== $k ====="
  ls -1 $k | head -200
  echo
  find $k -maxdepth 2 -type d -name "prec*" | sort | head -200
  echo
done