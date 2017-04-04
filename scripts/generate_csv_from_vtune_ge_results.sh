#!/bin/bash
start_result_num=433
end_result_num=702

generate_report_for_result() {
  result=r${1}ge
  #vtune="\"/Applications/Intel VTune Amplifier XE 2017.app/Contents/MacOS/amplxe-cl\""
  vtune="./amplxe-cl"
  outfile="$result/$result-hotspots.csv"
  if [[ ! -f $outfile ]]; then
    $vtune -report hotspots -r $result -format csv -csv-delimiter "|" -report-output $outfile
  fi
  outfile="$result/$result-hwevents.csv"
  if [[ ! -f $outfile ]]; then
    $vtune -report hw-events -r $result -format csv -csv-delimiter "|" -report-output $outfile
  fi
  outfile="$result/$result-summary.txt"
  if [[ ! -f $outfile ]]; then
    $vtune -report summary -r $result -report-output $outfile
  fi
}

generate_reports() {
  for f in $(seq -f "%03g" $start_result_num $end_result_num); do
    generate_report_for_result $f
  done
}

combine_hotspots_reports() {
  combined_csv=after_inlining_hotspots_combined.csv
  # Insert header
  orig_csv="r${start_result_num}ge/r${start_result_num}ge-hotspots.csv"
  head -n 1 $orig_csv | awk -F";" '{print "result|"$1}' > $combined_csv

  #Combine CSVs
  for f in $(seq -f "%03g" $start_result_num $end_result_num); do
    result=r${f}ge
    orig_csv=$result/$result-hotspots.csv
    if [[ ! -f $orig_csv ]]; then
      echo "WARNING: Could not find file $orig_csv. Skipping!"
    else
      grep "^obamadb::ml::dot" $orig_csv | awk -F";" -v f="$result" '{print f"|"$1}' >> $combined_csv
      grep "^obamadb::SVMTask::execute" $orig_csv | awk -F";" -v f="$result" '{print f"|"$1}' >> $combined_csv
    fi
  done
}

#combine_hwevents_reports() {
#  combined_csv=after_inlining_hwevents_combined.csv
#  # Insert header
#  orig_csv="r${start_result_num}ge/r${start_result_num}ge-hwevents.csv"
#  head -n 1 $orig_csv | awk -F";" '{print "result|"$1}' > $combined_csv
#
#  #Combine CSVs
#  for f in $(seq -f "%03g" $start_result_num $end_result_num); do
#    result=r${f}ge
#    orig_csv=$result/$result-hwevents.csv
#    grep "^obamadb::ml::dot" $orig_csv | awk -F";" -v f="$result" '{print f"|"$1}' >> $combined_csv
#    grep "^obamadb::SVMTask::execute" $orig_csv | awk -F";" -v f="$result" '{print f"|"$1}' >> $combined_csv
#  done
#}

combine_summary_reports() {
  combined_summary=after_inlining_summary_combined.csv
  echo "result executable t threads ne num_epochs nt num_trials tf train_file tf2 test_file" > $combined_summary
  for f in $(seq -f "%03g" $start_result_num $end_result_num); do
    result=r${f}ge
    orig_txt=$result/$result-summary.txt
    if [[ ! -f $orig_txt ]]; then
      echo "WARNING: Could not find file $orig_txt. Skipping!"
    else
      grep "Application Command Line" $orig_txt | awk -F":" -v f="$result" '{print f" "$2}' >> $combined_summary
    fi
  done
  cat $combined_summary | tr -s " " | tr " " "," > $combined_summary.tmp
  mv $combined_summary.tmp $combined_summary
}

combine_synth_cfg_files() {
  pushd ../data
  outfile="../Obamadb-osx/synth_cfg_files.csv"
  echo "filename,num_rows,num_features,density" > $outfile
  grep "" _synth_svm_m*.train.*tsv | tr -s " " | tr ":" "," | tr "\t" "," | tr " " "," >> $outfile
  popd
}

generate_epoch_times_from_log() {
  ./generate_epoch_times_csv_from_vtune_results.py > ../Obamadb-osx/after_inlining_epoch_times.csv
}

pushd ../ObamaDb-osx
generate_reports
combine_hotspots_reports
#combine_hwevents_reports
combine_summary_reports
popd

combine_synth_cfg_files
generate_epoch_times_from_log
