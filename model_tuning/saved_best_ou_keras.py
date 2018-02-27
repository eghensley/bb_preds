


scaler = StandardScaler()
def nn_model():
    model = Sequential()
    model.add(Dense(76, input_dim=39, kernel_initializer='normal', activation='elu'))
    model.add(Dropout(.45))
    model.add(Dense(50, kernel_initializer='normal', activation='elu'))
    model.add(Dropout(.45))
    model.add(Dense(25, kernel_initializer='normal', activation='elu'))
    model.add(Dropout(.15))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    return model 

for width in [6]:
        num_epochs = 120
        model = nn_model()
        for test_idx, train_idx in StratifiedShuffleSplit(n_splits=1, test_size=0.90, random_state=86).split(x_data, y_data):
            acc_results = []
            logloss_results = []
            history = model.fit(scaler.fit_transform(x_data.loc[train_idx]), np.ravel(y_data.loc[train_idx]), epochs=num_epochs, batch_size=16, verbose=1, validation_data=(scaler.fit_transform(x_data.loc[test_idx]), np.ravel(y_data.loc[test_idx])), shuffle = True)
            plt.plot(history.history['acc'], linestyle = '-.')
            plt.plot(history.history['val_acc'], linestyle = ':')
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test', 'validation'], loc='upper left')
            plt.show()
            print('accuracy graph ^')
            plt.plot(history.history['loss'], linestyle = '-.')
            plt.plot(history.history['val_loss'], linestyle = ':')            
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test', 'validation'], loc='upper left')
            plt.show()
            print('log loss graph ^')

            print("Results: best logloss %.4f @ epoch %s, best accuracy %.4f @ epoch %s" % (min(history.history['val_loss']), list(history.history['val_loss']).index(min(history.history['val_loss'])), max(history.history['val_acc']), list(history.history['val_acc']).index(max(history.history['val_acc']))))
            f = open('keras_model_tuning_ou.txt', 'a')
            f.write('BatchSize-%s_epochs-%s_model: \n best logloss %.4f @ epoch %s, best accuracy %.4f @ epoch %s\n' % (2**width, num_epochs, min(history.history['val_loss']), list(history.history['val_loss']).index(min(history.history['val_loss'])), max(history.history['val_acc']), list(history.history['val_acc']).index(max(history.history['val_acc']))))
            f.close()