// gru.js - TensorFlow.js GRU model for weather prediction

class WeatherGRUModel {
    constructor(numClasses, numFeatures) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.model = null;
        this.isTrained = false;
        this.buildModel();
    }

    buildModel() {
        // Clear any existing model
        if (this.model) {
            this.model.dispose();
        }

        this.model = tf.sequential({
            layers: [
                // Input shape: [null, 12, numFeatures] - [batch, timesteps, features]
                tf.layers.gru({
                    units: 64,
                    returnSequences: true,
                    inputShape: [12, this.numFeatures]
                }),
                tf.layers.dropout({ rate: 0.3 }),
                
                tf.layers.gru({
                    units: 32,
                    returnSequences: false
                }),
                tf.layers.dropout({ rate: 0.3 }),
                
                tf.layers.dense({
                    units: 16,
                    activation: 'relu'
                }),
                
                // Multi-output classification layer
                tf.layers.dense({
                    units: this.numClasses,
                    activation: 'softmax'
                })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        console.log('GRU model built successfully');
    }

    async train(X, y, options = {}) {
        const {
            epochs = 50,
            batchSize = 32,
            validationSplit = 0.2,
            callbacks = {}
        } = options;

        try {
            const history = await this.model.fit(X, y, {
                epochs,
                batchSize,
                validationSplit,
                shuffle: true,
                callbacks: callbacks
            });

            this.isTrained = true;
            return history;
        } catch (error) {
            console.error('Training error:', error);
            throw new Error(`Model training failed: ${error.message}`);
        }
    }

    predict(X) {
        if (!this.isTrained) {
            throw new Error('Model must be trained before making predictions');
        }
        return this.model.predict(X);
    }

    async evaluate(X, y) {
        if (!this.isTrained) {
            throw new Error('Model must be trained before evaluation');
        }

        const results = this.model.evaluate(X, y);
        const loss = await results[0].data();
        const accuracy = await results[1].data();

        // Clean up tensors
        results[0].dispose();
        results[1].dispose();

        return {
            loss: loss[0],
            accuracy: accuracy[0]
        };
    }

    async saveModel() {
        if (!this.isTrained) {
            throw new Error('No trained model to save');
        }

        const saveResult = await this.model.save('indexeddb://weather-gru-model');
        console.log('Model saved to IndexedDB');
        return saveResult;
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://weather-gru-model');
            this.isTrained = true;
            console.log('Model loaded from IndexedDB');
            return true;
        } catch (error) {
            console.warn('No saved model found:', error.message);
            return false;
        }
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }

    summary() {
        this.model.summary();
    }
}

// Export for use in other modules
export { WeatherGRUModel };