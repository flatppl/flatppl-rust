module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4, %5 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %6 = stablehlo.constant dense<9> : tensor<ui32>
    %7 = stablehlo.shift_right_logical %5, %6 : tensor<ui32>
    %8 = stablehlo.convert %7 : (tensor<ui32>) -> tensor<f32>
    %9 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %10 = stablehlo.multiply %8, %9 : tensor<f32>
    %11 = stablehlo.constant dense<2.0> : tensor<f32>
    %12 = stablehlo.constant dense<1.0> : tensor<f32>
    %13 = stablehlo.multiply %10, %11 : tensor<f32>
    %14 = stablehlo.subtract %13, %12 : tensor<f32>
    %15 = chlo.erf_inv %14 : tensor<f32> -> tensor<f32>
    %16 = stablehlo.constant dense<1.4142135> : tensor<f32>
    %17 = stablehlo.multiply %15, %16 : tensor<f32>
    %18 = stablehlo.multiply %17, %3 : tensor<f32>
    %19 = stablehlo.add %18, %2 : tensor<f32>
    %20 = stablehlo.multiply %1, %19 : tensor<f32>
    %21 = stablehlo.add %0, %20 : tensor<f32>
    %22 = stablehlo.exponential %21 : tensor<f32>
    return %22, %4 : tensor<f32>, tensor<2xui64>
  }
}
