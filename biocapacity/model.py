import keras
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense
from keras.applications import vgg16
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class EuroSatRGBClassifier:
    def __init__(self, input_shape, num_classes, model_file):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_file = model_file
        self.model = self.build_model()
        self.callbacks = self.define_callbacks()


    def define_callbacks(self):
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True
        )
        
        reduce_LR = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5
        )
        
        checkpoint = ModelCheckpoint(
            filepath=self.model_file,
            monitor='val_loss',
            save_best_only=True
        )
        
        return [reduce_LR, early_stop, checkpoint]
    
    def build_model(self):
        conv_base = vgg16.VGG16(include_top=False, input_shape=self.input_shape)
        model = Sequential([
            conv_base,
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        optimizer = RMSprop(lr=1e-4)
                
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', optimizer.lr]
        )
        return model
    
    def train(self,epochs, X_train,y_train, X_test, y_test, batch_size):
        train_data_generator = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True)

        test_data_generator = ImageDataGenerator()
        
        train_generator = train_data_generator.flow(
            X_train,
            y_train,
            batch_size = batch_size,
            seed = 42
        )

        test_generator = test_data_generator.flow(
            X_test,
            y_test,
            batch_size = batch_size,
            seed = 42
        )
        
        history = self.model.fit_generator(train_generator,
                 steps_per_epoch = len(X_train) // batch_size,
                 epochs = epochs,
                 validation_data = test_generator,
                 validation_steps = len(X_test) // batch_size,
                 callbacks = self.callback_list)
        
        return history