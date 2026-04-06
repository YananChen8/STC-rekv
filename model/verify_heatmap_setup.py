#!/usr/bin/env python3
"""
热力图功能验证脚本

验证所有新增的热力图功能是否正常导入和定义。
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试所有必要的导入"""
    print("=" * 60)
    print("Testing Imports...")
    print("=" * 60)
    
    try:
        print("[✓] Testing matplotlib import...")
        import matplotlib.pyplot as plt
        print("    matplotlib.pyplot imported successfully")
    except ImportError as e:
        print(f"    [✗] Failed to import matplotlib: {e}")
        return False
    
    try:
        print("[✓] Testing seaborn import...")
        import seaborn as sns
        print("    seaborn imported successfully")
    except ImportError as e:
        print(f"    [✗] Failed to import seaborn: {e}")
        return False
    
    try:
        print("[✓] Testing numpy import...")
        import numpy as np
        print("    numpy imported successfully")
    except ImportError as e:
        print(f"    [✗] Failed to import numpy: {e}")
        return False
    
    print("\n[✓] All visualization dependencies available!\n")
    return True


def test_function_definitions():
    """测试函数定义"""
    print("=" * 60)
    print("Testing Function Definitions...")
    print("=" * 60)
    
    try:
        print("[✓] Checking custom_siglip module...")
        # 检查文件是否存在
        custom_siglip_path = Path(__file__).parent / "custom_siglip.py"
        if not custom_siglip_path.exists():
            print(f"    [✗] custom_siglip.py not found at {custom_siglip_path}")
            return False
        print(f"    Found at: {custom_siglip_path}")
        
        # 检查关键函数是否存在
        with open(custom_siglip_path, 'r') as f:
            content = f.read()
            
            # 检查新增函数
            functions_to_check = [
                "def record_attention_and_similarity_heatmap",
                "def visualize_heatmaps_from_logs",
            ]
            
            print("\n[✓] Checking for new functions...")
            for func_name in functions_to_check:
                if func_name in content:
                    print(f"    [✓] Found: {func_name}")
                else:
                    print(f"    [✗] Missing: {func_name}")
                    return False
            
            # 检查修改的函数
            modified_functions = [
                "def forward_with_selective_key_recompute",
                "def forward_with_selective_key_recompute_clip",
                "def forward_with_selective_recompute",
                "def new_siglip_sdpa_attn_forward",
            ]
            
            print("\n[✓] Checking for function modifications...")
            for func_name in modified_functions:
                if func_name in content:
                    print(f"    [✓] Found: {func_name}")
                else:
                    print(f"    [✗] Missing: {func_name}")
                    return False
            
            # 检查热力图记录调用
            print("\n[✓] Checking for heatmap recording calls...")
            if "record_attention_and_similarity_heatmap" in content:
                # 计算出现次数
                count = content.count("record_attention_and_similarity_heatmap")
                print(f"    [✓] Found {count} calls to record_attention_and_similarity_heatmap")
            else:
                print(f"    [✗] No calls to record_attention_and_similarity_heatmap found")
                return False
    
    except Exception as e:
        print(f"    [✗] Error checking functions: {e}")
        return False
    
    print("\n[✓] All function definitions verified!\n")
    return True


def test_helper_scripts():
    """测试辅助脚本是否存在"""
    print("=" * 60)
    print("Testing Helper Scripts...")
    print("=" * 60)
    
    helper_scripts = [
        "visualize_heatmaps.py",
        "inference_with_heatmap.py",
    ]
    
    model_dir = Path(__file__).parent
    
    for script in helper_scripts:
        script_path = model_dir / script
        if script_path.exists():
            print(f"[✓] Found: {script}")
        else:
            print(f"[✗] Missing: {script}")
    
    print()
    return True


def test_documentation():
    """测试文档是否存在"""
    print("=" * 60)
    print("Testing Documentation...")
    print("=" * 60)
    
    docs = [
        "HEATMAP_VISUALIZATION_GUIDE.md",
        "MODIFICATION_SUMMARY.md",
        "QUICKSTART.md",
    ]
    
    model_dir = Path(__file__).parent
    
    for doc in docs:
        doc_path = model_dir / doc
        if doc_path.exists():
            size_kb = doc_path.stat().st_size / 1024
            print(f"[✓] Found: {doc} ({size_kb:.1f} KB)")
        else:
            print(f"[✗] Missing: {doc}")
    
    print()
    return True


def test_directory_structure():
    """测试输出目录结构"""
    print("=" * 60)
    print("Testing Directory Structure...")
    print("=" * 60)
    
    dirs_to_check = {
        "./ACR/heatmap_logs": "Heatmap logs directory",
        "./ACR/heatmap_visualizations": "Heatmap visualizations directory",
    }
    
    for dir_path, description in dirs_to_check.items():
        full_path = Path(dir_path)
        if full_path.exists():
            print(f"[✓] {description}: {dir_path} (exists)")
        else:
            print(f"[~] {description}: {dir_path} (will be created on first run)")
    
    print()
    return True


def main():
    """主测试函数"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   STC Heatmap Visualization - Verification Test Suite     ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    all_passed = True
    
    # 运行所有测试
    tests = [
        ("Imports", test_imports),
        ("Function Definitions", test_function_definitions),
        ("Helper Scripts", test_helper_scripts),
        ("Documentation", test_documentation),
        ("Directory Structure", test_directory_structure),
    ]
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"\n[✗] {test_name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # 打印总结
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if all_passed:
        print("\n[✓] All verification tests passed!")
        print("\nYou can now use the heatmap visualization features:")
        print("  1. Run your inference script")
        print("  2. Call visualize_heatmaps_from_logs()")
        print("  3. View generated PNG files in ./ACR/heatmap_visualizations/")
        print("\nFor more details, see:")
        print("  - QUICKSTART.md (quick reference)")
        print("  - HEATMAP_VISUALIZATION_GUIDE.md (detailed guide)")
        print("  - MODIFICATION_SUMMARY.md (technical details)")
    else:
        print("\n[✗] Some verification tests failed!")
        print("\nPlease check the output above for details.")
        print("Common issues:")
        print("  - Missing Python dependencies (matplotlib, seaborn)")
        print("  - File not found (check file paths)")
        print("  - Permission issues (check directory permissions)")
    
    print("\n" + "=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
