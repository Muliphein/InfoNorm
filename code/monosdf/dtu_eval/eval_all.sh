# python evaluate_single_scene.py --input_mesh result/24/neuris.ply --scan_id 24 --output_dir neuris_24 --DTU ./dtu
# python evaluate_single_scene.py --input_mesh result/37/neuris.ply --scan_id 37 --output_dir neuris_37 --DTU ./dtu
# python evaluate_single_scene.py --input_mesh result/40/neuris.ply --scan_id 40 --output_dir neuris_40 --DTU ./dtu

# python evaluate_single_scene.py --input_mesh result/24/neuris+.ply --scan_id 24 --output_dir neuris+_24 --DTU ./dtu
# python evaluate_single_scene.py --input_mesh result/37/neuris+.ply --scan_id 37 --output_dir neuris+_37 --DTU ./dtu
# python evaluate_single_scene.py --input_mesh result/40/neuris+.ply --scan_id 40 --output_dir neuris+_40 --DTU ./dtu


python evaluate_single_scene.py --input_mesh result/24/monosdf_align.ply --scan_id 24 --output_dir monosdf_24 --DTU ./dtu
python evaluate_single_scene.py --input_mesh result/37/monosdf_align.ply --scan_id 37 --output_dir monosdf_37 --DTU ./dtu
python evaluate_single_scene.py --input_mesh result/40/monosdf_align.ply --scan_id 40 --output_dir monosdf_40 --DTU ./dtu

python evaluate_single_scene.py --input_mesh result/24/monosdf+_align.ply --scan_id 24 --output_dir monosdf+_24 --DTU ./dtu
python evaluate_single_scene.py --input_mesh result/37/monosdf+_align.ply --scan_id 37 --output_dir monosdf+_37 --DTU ./dtu
python evaluate_single_scene.py --input_mesh result/40/monosdf+_align.ply --scan_id 40 --output_dir monosdf+_40 --DTU ./dtu