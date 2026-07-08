module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<3.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3, %4 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %5 = stablehlo.constant dense<9> : tensor<ui32>
    %6 = stablehlo.shift_right_logical %4, %5 : tensor<ui32>
    %7 = stablehlo.convert %6 : (tensor<ui32>) -> tensor<f32>
    %8 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %9 = stablehlo.multiply %7, %8 : tensor<f32>
    %10 = stablehlo.log %9 : tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %12 = stablehlo.divide %2, %0 : tensor<f32>
    %13 = stablehlo.power %11, %12 : tensor<f32>
    %14 = stablehlo.multiply %1, %13 : tensor<f32>
    return %14, %3 : tensor<f32>, tensor<2xui64>
  }
}
