package org.example.model;

import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealMatrix;
import org.example.ParameterContainer;

import java.util.ArrayList;
import java.util.List;

public class OptimizerAdam {
    private final ParameterContainer model;
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    // Списки для хранения матриц для каждого параметра
    private List<RealMatrix> momentum;  // нужен для проверки, если находимся в седле, проверить потенциальное движение нашего пунктира в другие стороны
    private List<RealMatrix> rmsPropagation;  // Root Mean Square propagation - масштабирование параметра весов, если веса слишком большие (высокая скорость обучения), этот параметр их уменьшит, также работает и в обратную сторону. Но зачем это, если можно наоборот "отсечь" ненужные параметры и сделать акцент на важных, или тогда уменьшится эмбеддинг, то есть эти эмбеддинги перестанут быть уникальными?
    private int t;  // Счетчик шагов

    public OptimizerAdam(ParameterContainer model, double learningRate, double beta1, double beta2, double epsilon) {
        this.model = model;
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        this.t = 0;  // начальный шаг и хранилища матриц
        this.momentum = new ArrayList<>();
        this.rmsPropagation = new ArrayList<>();

        // Хитро, затратно, гениально: инициализация матриц-параметров своей размерности нулевыми значениями
        for (RealMatrix param : model.getParameters()) {
            momentum.add(param.copy().scalarMultiply(0.0));
            rmsPropagation.add(param.copy().scalarMultiply(0.0));
        }
    }

    public OptimizerAdam(ParameterContainer model, double learningRate) {
        this(model, learningRate, 0.9, 0.999, 1e-8);
    }

    public void step() {
        this.t++;

        List<RealMatrix> params = model.getParameters();
        List<RealMatrix> grads = model.getGradients();

        if (params.size() != grads.size()) {
            throw new IllegalStateException("OptimizerAdam.java: Количество параметров не совпадает с числом градиентов");
        }

        for (int i = 0; i < grads.size(); i++) {
            RealMatrix param = params.get(i);
            RealMatrix grad = grads.get(i);
            RealMatrix momentum_prev = this.momentum.get(i);  // Заметили? Первые значения в momentum и rmsProp всегда нулевые!
            RealMatrix rmsPropagation_prev = this.rmsPropagation.get(i);

            // Логика Adam - улучшенного градиентного спуска, в котором w_1_new = w_1 - educationSpeed * df(...) / dw_1, скорость обучения "ню" задается ручками:

            // Расчет скользящего среднего - моментума и запоминание состояния в списке momentum:
            RealMatrix momentum_t = momentum_prev.scalarMultiply(beta1).add(grad.scalarMultiply(1.0 - beta1));
            this.momentum.set(i, momentum_t);

            // Обновление квадратов градиентов и сохранение состояния в rmsPropagation:
            RealMatrix grad_squared = elementWisePower(grad, 2.0);
            RealMatrix rmsPropagation_t = rmsPropagation_prev.scalarMultiply(beta2).add(grad_squared.scalarMultiply(1.0 - beta2));
            this.rmsPropagation.set(i, rmsPropagation_t);


            // коррекция смещения для параметров momentum_i и rmsPropagation_i
            RealMatrix momentum_hat = momentum_t.scalarMultiply(1.0 / (1.0 - Math.pow(beta1, t)));
            RealMatrix rmsPropagation_hat = rmsPropagation_t.scalarMultiply(1.0 / (1.0 - Math.pow(beta2, t)));

            RealMatrix rmsPropagation_hat_sqrt = elementWiseSqrt(rmsPropagation_hat);
            RealMatrix denominator = rmsPropagation_hat_sqrt.scalarAdd(epsilon);
            RealMatrix update = elementWiseDivide(momentum_hat, denominator).scalarMultiply(learningRate);

            // Обновление параметра
            /*
            param.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {

                @Override
                public double visit(int row, int column, double value) {
                    return value - update.getEntry(row, column);
                }
            });
            */
            RealMatrix param_new = param.subtract(update);
            param.setSubMatrix(param_new.getData(), 0, 0);
         }
    }

    public void zeroGrad() {
        model.zeroGradients();
    }

    // Описание вспомогательных методов для расчета (пиздец, господа, копайтесь в интерфейсах RealMatrix)):
    // Использование анонимных классов для переопределения visit
    private RealMatrix elementWisePower(RealMatrix matrix, double power) {
        RealMatrix result = matrix.copy();
        result.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor(){
            @Override
            public double visit(int row, int column, double value) {
                return Math.pow(value, power);
            }
        });
        return result;
    }

    private RealMatrix elementWiseSqrt(RealMatrix matrix) {
        RealMatrix result = matrix.copy();
        result.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor(){
            @Override
            public double visit(int row, int column, double value) {
                return Math.sqrt(value);
            }
        });
        return result;
    }

    private RealMatrix elementWiseDivide(RealMatrix numerator, RealMatrix denominator) {
        RealMatrix result = numerator.copy();
        result.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor(){
            @Override
            public double visit(int row, int column, double value) {
                return value / denominator.getEntry(row, column);
            }
        });
        return result;
    }
}
