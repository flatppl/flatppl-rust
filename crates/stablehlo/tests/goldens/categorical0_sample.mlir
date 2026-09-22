module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3, %4 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %5 = stablehlo.constant dense<9> : tensor<ui32>
    %6 = stablehlo.shift_right_logical %4, %5 : tensor<ui32>
    %7 = stablehlo.convert %6 : (tensor<ui32>) -> tensor<f32>
    %8 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %9 = stablehlo.multiply %7, %8 : tensor<f32>
    %12 = stablehlo.constant dense<0.20000000298023224> : tensor<f32>
    %13 = stablehlo.compare LT, %12, %9 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %14 = stablehlo.select %13, %2, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %15 = stablehlo.add %1, %14 : tensor<f32>
    %18 = stablehlo.constant dense<0.5> : tensor<f32>
    %19 = stablehlo.compare LT, %18, %9 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %20 = stablehlo.select %19, %2, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %21 = stablehlo.add %15, %20 : tensor<f32>
    return %21, %3 : tensor<f32>, tensor<2xui64>
  }
}
