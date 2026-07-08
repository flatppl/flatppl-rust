module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3, %4 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %5 = stablehlo.constant dense<9> : tensor<ui32>
    %6 = stablehlo.shift_right_logical %4, %5 : tensor<ui32>
    %7 = stablehlo.convert %6 : (tensor<ui32>) -> tensor<f32>
    %8 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %9 = stablehlo.multiply %7, %8 : tensor<f32>
    %10 = stablehlo.compare LT, %9, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %11 = stablehlo.select %10, %2, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %11, %3 : tensor<f32>, tensor<2xui64>
  }
}
