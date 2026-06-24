import numpy as np
import torch


def compare_recursive(val1, val2, key_name="root"):
    # 1. 타입이 다른 경우
    if type(val1) != type(val2):
        print(
            f"❌ '{key_name}'의 타입이 일치하지 않습니다. ({type(val1)} vs {type(val2)})"
        )
        return False

    # 2. 딕셔너리(dict)인 경우
    if isinstance(val1, dict):
        if val1.keys() != val2.keys():
            print(f"❌ '{key_name}' 내부의 키 구성이 다릅니다.")
            return False
        for k in val1.keys():
            if not compare_recursive(val1[k], val2[k], f"{key_name} -> {k}"):
                return False
        return True

    # 3. 리스트(list) 또는 튜플(tuple)인 경우
    elif isinstance(val1, (list, tuple)):
        if len(val1) != len(val2):
            print(f"❌ '{key_name}'의 길이가 다릅니다.")
            return False
        for i in range(len(val1)):
            if not compare_recursive(val1[i], val2[i], f"{key_name}[{i}]"):
                return False
        return True

    # 4. 파이토치 텐서(Tensor)인 경우
    elif torch.is_tensor(val1):
        if not torch.equal(val1, val2):
            print(f"❌ '{key_name}' 텐서 값이 다릅니다.")
            return False
        return True

    # 5. 넘파이 배열(Numpy array)인 경우
    elif isinstance(val1, np.ndarray):
        if not np.array_equal(val1, val2):
            print(f"❌ '{key_name}' (Numpy) 값이 다릅니다.")
            return False
        return True

    # 6. 기타 기본 타입 (int, float, str 등)
    else:
        try:
            if val1 != val2:
                print(f"❌ '{key_name}' 값이 다릅니다.")
                return False
        except Exception:
            # 비교 연산이 실패하는 특수한 경우를 위해 문자열 비교로 대체
            if str(val1) != str(val2):
                print(f"❌ '{key_name}' 데이터가 다릅니다.")
                return False
        return True


def compare_models(path1, path2):
    print("🔍 비교를 시작합니다... (용량이 크면 시간이 조금 걸릴 수 있습니다)")
    try:
        model1_dict = torch.load(path1, map_location="cpu", weights_only=False)
        model2_dict = torch.load(path2, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ 파일을 불러오는 중 오류 발생: {e}")
        return

    if compare_recursive(model1_dict, model2_dict):
        print("\n✅ [확인 완료] 두 모델의 모든 데이터가 100% 일치합니다!")
    else:
        print("\n❌ [확인 결과] 두 모델은 서로 다른 데이터가 포함되어 있습니다.")


# 경로 설정
path1 = r"C:\python\최종 논문 용\람다랩스로 돌린거 최종모델\output\MoCo_seed87\SAE\stage5_out_d4096_gated_sp3200.0_aux0.03125_tied_ep008.pt"
path2 = (
    r"C:\Users\admin\Downloads\stage5_out_d4096_gated_sp3200.0_aux0.03125_tied_ep008.pt"
)

compare_models(path1, path2)
