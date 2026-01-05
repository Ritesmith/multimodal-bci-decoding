import sys
sys.path.insert(0, 'D:/Stazica/my_files/University_works/self study/脑机接口课程遐想')

try:
    import main
    print("main.py imported successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()