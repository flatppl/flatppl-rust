module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2, %3 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %4 = stablehlo.constant dense<9> : tensor<ui32>
    %5 = stablehlo.shift_right_logical %3, %4 : tensor<ui32>
    %6 = stablehlo.convert %5 : (tensor<ui32>) -> tensor<f32>
    %7 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %8 = stablehlo.multiply %6, %7 : tensor<f32>
    %9 = stablehlo.constant dense<-1.0> : tensor<f32>
    %10 = stablehlo.divide %9, %0 : tensor<f32>
    %11 = stablehlo.power %8, %10 : tensor<f32>
    %12 = stablehlo.multiply %1, %11 : tensor<f32>
    return %12, %2 : tensor<f32>, tensor<2xui64>
  }
}
