module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0, %1 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %2 = stablehlo.constant dense<9> : tensor<ui32>
    %3 = stablehlo.shift_right_logical %1, %2 : tensor<ui32>
    %4 = stablehlo.convert %3 : (tensor<ui32>) -> tensor<f32>
    %5 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %6 = stablehlo.multiply %4, %5 : tensor<f32>
    %7 = stablehlo.constant dense<-1.0> : tensor<f32>
    %8 = stablehlo.constant dense<4.0> : tensor<f32>
    %9 = stablehlo.multiply %8, %6 : tensor<f32>
    %10 = stablehlo.add %7, %9 : tensor<f32>
    return %10, %0 : tensor<f32>, tensor<2xui64>
  }
}
