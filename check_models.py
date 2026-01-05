import sys
sys.path.insert(0, 'D:/Stazica/my_files/University_works/self study/脑机接口课程遐想')

try:
    from src.models import MultiModalFusionModel, TORCH_AVAILABLE
    print('Models import successful, TORCH_AVAILABLE:', TORCH_AVAILABLE)
except Exception as e:
    print('Models import failed:', e)
    import traceback
    traceback.print_exc()