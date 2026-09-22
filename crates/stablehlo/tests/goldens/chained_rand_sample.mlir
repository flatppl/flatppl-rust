module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2, %3 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %4 = stablehlo.constant dense<9> : tensor<ui32>
    %5 = stablehlo.shift_right_logical %3, %4 : tensor<ui32>
    %6 = stablehlo.convert %5 : (tensor<ui32>) -> tensor<f32>
    %7 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %8 = stablehlo.multiply %6, %7 : tensor<f32>
    %9 = stablehlo.constant dense<2.0> : tensor<f32>
    %13 = stablehlo.constant dense<1.4142135> : tensor<f32>
    %17, %18 = stablehlo.rng_bit_generator %2, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %19 = stablehlo.constant dense<9> : tensor<ui32>
    %20 = stablehlo.shift_right_logical %18, %19 : tensor<ui32>
    %21 = stablehlo.convert %20 : (tensor<ui32>) -> tensor<f32>
    %22 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %23 = stablehlo.multiply %21, %22 : tensor<f32>
    %24 = stablehlo.multiply %23, %9 : tensor<f32>
    %25 = stablehlo.subtract %24, %1 : tensor<f32>
    %26 = chlo.erf_inv %25 : tensor<f32> -> tensor<f32>
    %27 = stablehlo.constant dense<1.4142135> : tensor<f32>
    %28 = stablehlo.multiply %26, %27 : tensor<f32>
    %29 = stablehlo.multiply %1, %28 : tensor<f32>
    %30 = stablehlo.add %1, %29 : tensor<f32>
    return %30, %17 : tensor<f32>, tensor<2xui64>
  }
}
