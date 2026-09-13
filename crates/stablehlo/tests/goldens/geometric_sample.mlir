module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %2, %3 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %4 = stablehlo.constant dense<9> : tensor<ui32>
    %5 = stablehlo.shift_right_logical %3, %4 : tensor<ui32>
    %6 = stablehlo.convert %5 : (tensor<ui32>) -> tensor<f32>
    %7 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %8 = stablehlo.multiply %6, %7 : tensor<f32>
    %9 = stablehlo.log %8 : tensor<f32>
    %11 = stablehlo.constant dense<-0.3566749691963196> : tensor<f32>
    %12 = stablehlo.divide %9, %11 : tensor<f32>
    %13 = stablehlo.floor %12 : tensor<f32>
    return %13, %2 : tensor<f32>, tensor<2xui64>
  }
}
