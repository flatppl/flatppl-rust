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
    %10 = stablehlo.subtract %2, %1 : tensor<f32>
    %11 = stablehlo.multiply %9, %10 : tensor<f32>
    %12 = stablehlo.add %11, %1 : tensor<f32>
    %13 = stablehlo.log %12 : tensor<f32>
    %14 = stablehlo.subtract %2, %0 : tensor<f32>
    %15 = stablehlo.log %14 : tensor<f32>
    %16 = stablehlo.divide %13, %15 : tensor<f32>
    %17 = stablehlo.floor %16 : tensor<f32>
    return %17, %3 : tensor<f32>, tensor<2xui64>
  }
}
