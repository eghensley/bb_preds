class VotingClassifier():
    def __init__(self, models, weights = False, req_feats = False):
        self._stored_models = {}
        if not weights:
            weights = [float(1)/float(len(models)) for i in len(models)]
        if not req_feats:
            req_feats = [False for i in len(models)]
        for i, (model_name, model_weight, model_feats) in enumerate(zip(models, weights, req_feats)):
            self._stored_models[i] = {'model': model_name, 'weight': model_weight, 'features': model_feats}
            
    def predict_proba(self, x):   
        all_preds = []
        for i in range(len(x)):
            ind_x = x.iloc[i]
            ind_1 = []
            ind_2 = []
            for each in self._stored_models.values():
                pred = each['model'].predict_proba(ind_x[each['features']].values.reshape(1,-1))     
                ind_1.append(pred[0][0] * each['weight'])
                ind_2.append(pred[0][1] * each['weight'])
            all_preds.append([ind_1, ind_2])        
        return all_preds
  
    def predict(self, x):   
        all_preds = []
        for i in range(len(x)):
            ind_x = x.iloc[i]
            ind_1 = []
            ind_2 = []
            for each in self._stored_models.values():
                pred = each['model'].predict_proba(ind_x[each['features']].values.reshape(1,-1))     
                ind_1.append(pred[0][0] * each['weight'])
                ind_2.append(pred[0][1] * each['weight'])
            if ind_1 > ind_2:
                all_preds.append(0)        
            else:
                all_preds.append(1)
        return all_preds