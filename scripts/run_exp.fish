#!/usr/bin/env fish


for sweep in 5.0 4.6 4.2 3.8 3.4 3.0 2.6 2.2 1.8 1.4 1.0 0.6 0.2 0.01
  for  meas in 5.0 4.6 4.2 3.8 3.4 3.0 2.6 2.2 1.8 1.4 1.0 0.6 0.2 0.01
    for iato in 5.0 4.6 4.2 3.8 3.4 3.0 2.6 2.2 1.8 1.4 1.0 0.6 0.2 0.01
      set -x wait (math "20*(9*9*$sweep/1000+$iato/1000)/60+0.5")
      echo "----------------------------------------------------------"
      echo (math $sweep/1000) (math $meas/1000) (math $iato/1000) "bp_""$sweep""_mp_""$meas""_id_""$iato" "duration "$wait"m"
      echo "----------------------------------------------------------"
      timeout "$wait"m /home/user/gr-mmwave_beam_control/examples/complete_initial_access.py --beam-period (math $sweep/1000) --meas-period (math $meas/1000) --ia-interval (math $iato/1000) --file-suffix "bp_""$sweep""_mp_""$meas""_id_""$iato"
    end
  end
end

mv $HOME/*.log /home/user/very_low_long_2/
