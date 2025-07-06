#!/usr/bin/env python3
"""
Week 3 Complete Test Suite
Runs all test parts for the Module system implementation.

Test Structure:
- Part 1: Core Module System (module registration, state management, repr)
- Part 2: Simplest Concrete Modules (Linear, Tanh)
- Part 3: Initialization Functions (xavier, kaiming, etc.)
- Part 4: Concrete Modules (activations, containers, layers, loss functions)
"""

import sys
import os
import subprocess
from pathlib import Path

def run_test_part(part_name, part_file):
    """Run a specific test part and capture results"""
    print(f"\n{'='*60}")
    print(f"RUNNING {part_name}")
    print(f"{'='*60}")
    
    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, part_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Check if the test passed
        if result.returncode == 0:
            print(f"✓ {part_name} COMPLETED SUCCESSFULLY")
            return True
        else:
            print(f"✗ {part_name} FAILED (return code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {part_name} TIMED OUT (>5 minutes)")
        return False
    except Exception as e:
        print(f"✗ {part_name} ERROR: {e}")
        return False

def main():
    """Run all Week 3 tests"""
    print("CLOWNPIECE WEEK 3 - COMPLETE TEST SUITE")
    print("Module System Implementation Tests")
    print(f"Working directory: {os.getcwd()}")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test parts in order
    test_parts = [
        ("PART 1: Core Module System", "grade_part1.py"),
        ("PART 2: Simplest Concrete Modules", "grade_part2.py"),
        ("PART 3: Initialization Functions", "grade_part3.py"),
        ("PART 4: Concrete Modules", "grade_part4.py"),
    ]
    
    results = {}
    
    # Run each test part
    for part_name, part_file in test_parts:
        # Use absolute path relative to script directory
        full_part_path = os.path.join(script_dir, part_file)
        if not os.path.exists(full_part_path):
            print(f"✗ {part_name} - FILE NOT FOUND: {full_part_path}")
            results[part_name] = False
        else:
            results[part_name] = run_test_part(part_name, full_part_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("WEEK 3 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(test_parts)
    
    for part_name, part_file in test_parts:
        status = "✓ PASSED" if results[part_name] else "✗ FAILED"
        print(f"{part_name:<40} {status}")
        if results[part_name]:
            passed += 1
    
    print(f"\nOVERALL RESULT: {passed}/{total} parts passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Week 3 implementation is complete.")
        return 0
    else:
        print(f"❌ {total - passed} test parts failed. Please review the failing parts.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
