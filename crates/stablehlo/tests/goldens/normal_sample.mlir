module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2, %3 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %4 = stablehlo.constant dense<9> : tensor<ui32>
    %5 = stablehlo.shift_right_logical %3, %4 : tensor<ui32>
    %6 = stablehlo.convert %5 : (tensor<ui32>) -> tensor<f32>
    %7 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %8 = stablehlo.multiply %6, %7 : tensor<f32>
    %9 = stablehlo.constant dense<2.0> : tensor<f32>
    %10 = stablehlo.multiply %8, %9 : tensor<f32>
    %11 = stablehlo.subtract %10, %1 : tensor<f32>
    %12 = chlo.erf_inv %11 : tensor<f32> -> tensor<f32>
    %13 = stablehlo.constant dense<1.4142135> : tensor<f32>
    %14 = stablehlo.multiply %12, %13 : tensor<f32>
    %15 = stablehlo.multiply %1, %14 : tensor<f32>
    %16 = stablehlo.add %0, %15 : tensor<f32>
    return %16, %2 : tensor<f32>, tensor<2xui64>
  }
}
