#!/bin/bash

# Find and rename precision settings files

find . -type f -name "precision_settings_*.json" | while read -r file; do
    dir=$(dirname "$file")
    base=$(basename "$file")

    # Rename according to pattern
    case "$base" in
        precision_settings_1.json)
            mv "$file" "$dir/prec_setting_1.json"
            echo "Renamed: $file → $dir/prec_setting_1.json"
            ;;
        precision_settings_2.json)
            mv "$file" "$dir/prec_setting_2.json"
            echo "Renamed: $file → $dir/prec_setting_2.json"
            ;;
        precision_settings_3.json)
            mv "$file" "$dir/prec_setting_3.json"
            echo "Renamed: $file → $dir/prec_setting_3.json"
            ;;
        precision_settings_4.json)
            mv "$file" "$dir/prec_setting_4.json"
            echo "Renamed: $file → $dir/prec_setting_4.json"
            ;;
    esac
done

echo "Rename complete!"
