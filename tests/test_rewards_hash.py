import os

def test_rewards_hash_placeholder_present():
    # Verify the placeholder string is present in the source files
    local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    grade_path = os.path.join(local_dir, "src", "protean", "tasks", "elementwise_add_relu", "donotaccess", "grade.py")
    calibrate_path = os.path.join(local_dir, "train", "calibrate.py")
    
    placeholder = "<sha256 baked at image build>"
    
    assert os.path.exists(grade_path), f"grade.py not found at {grade_path}"
    assert os.path.exists(calibrate_path), f"calibrate.py not found at {calibrate_path}"
    
    with open(grade_path, "r", encoding="utf-8") as f:
        grade_content = f.read()
    assert placeholder in grade_content, f"Placeholder '{placeholder}' not found in {grade_path}"
    
    with open(calibrate_path, "r", encoding="utf-8") as f:
        calibrate_content = f.read()
    assert placeholder in calibrate_content, f"Placeholder '{placeholder}' not found in {calibrate_path}"
