module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1, %2 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %3 = stablehlo.constant dense<9> : tensor<ui32>
    %4 = stablehlo.shift_right_logical %2, %3 : tensor<ui32>
    %5 = stablehlo.convert %4 : (tensor<ui32>) -> tensor<f32>
    %6 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %7 = stablehlo.multiply %5, %6 : tensor<f32>
    %8 = stablehlo.log %7 : tensor<f32>
    %9 = stablehlo.constant dense<-2.0> : tensor<f32>
    %10 = stablehlo.multiply %9, %8 : tensor<f32>
    %11 = stablehlo.sqrt %10 : tensor<f32>
    %12 = stablehlo.multiply %0, %11 : tensor<f32>
    return %12, %1 : tensor<f32>, tensor<2xui64>
  }
}
