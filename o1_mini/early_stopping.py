class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): 개선이 없을 때 기다릴 에포크 수
            verbose (bool): 로그 출력 여부
            delta (float): 개선으로 인정할 최소 변화량
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc, model, save_path):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, save_path)
            self.counter = 0

    def save_checkpoint(self, model, save_path):
        '''Validation 결과가 개선되면 모델 상태를 저장'''
        torch.save(model.state_dict(), save_path)
        if self.verbose:
            print(f'Validation 정확도 향상. 모델 저장: {save_path}')