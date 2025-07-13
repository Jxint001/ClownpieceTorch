#!/usr/bin/env python3

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graderlib import grader_summary, print_separate_line
import grade_part1
import grade_part2  
import grade_part3

def main():
    print_separate_line()
    print("Running Week 4 Comprehensive Tests")
    print_separate_line()
    
    print("\n=== Part 1: Dataset, Dataloader, and Basic Optimizers ===")
    grade_part1.test_csvdataset_basic()
    grade_part1.test_csvdataset_transform()
    grade_part1.test_imagedataset_basic()
    grade_part1.test_imagedataset_transform()
    grade_part1.test_dataloader_batching()
    grade_part1.test_dataloader_shuffle()
    grade_part1.test_dataloader_drop_last()
    grade_part1.test_transforms()
    grade_part1.test_sgd_vanilla()
    grade_part1.test_adam_step()
    grade_part1.test_optimizer_add_param_group()
    grade_part1.test_optimizer_zero_grad()
    grade_part1.test_lambdalr()
    grade_part1.test_explr()
    grade_part1.test_steplr()
    grade_part1.test_scheduler_get_last_lr()
    grade_part1.test_scheduler_last_epoch()
    
    print("\n=== Part 2: Advanced Optimizer Features ===")
    grade_part2.test_sgd_momentum_weight_decay()
    grade_part2.test_adam_bias_correction()
    grade_part2.test_sgd_momentum_accumulation()
    grade_part2.test_adam_weight_decay()
    grade_part2.test_adam_epsilon()
    
    print("\n=== Part 3: Advanced Learning Rate Scheduler Features ===")
    grade_part3.test_lambdalr_custom()
    grade_part3.test_explr_half()
    grade_part3.test_steplr_3()
    grade_part3.test_lambdalr_multiple_groups()
    grade_part3.test_explr_different_gamma()
    grade_part3.test_steplr_different_steps()
    grade_part3.test_scheduler_explicit_epoch()
    
    print("\n=== Final Summary ===")
    print(f"Total tests: 29 test cases, 290 points total")
    grader_summary()

if __name__ == "__main__":
    main()
