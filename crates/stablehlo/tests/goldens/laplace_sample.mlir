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
    %11 = stablehlo.subtract %3, %2 : tensor<f32>
    %12 = stablehlo.multiply %10, %11 : tensor<f32>
    %13 = stablehlo.add %12, %2 : tensor<f32>
    %14 = stablehlo.constant dense<0.5> : tensor<f32>
    %15 = stablehlo.subtract %13, %14 : tensor<f32>
    %16 = stablehlo.compare GE, %15, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %17 = stablehlo.constant dense<1.0> : tensor<f32>
    %18 = stablehlo.constant dense<-1.0> : tensor<f32>
    %19 = stablehlo.select %16, %17, %18 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.abs %15 : tensor<f32>
    %21 = stablehlo.constant dense<2.0> : tensor<f32>
    %22 = stablehlo.multiply %21, %20 : tensor<f32>
    %23 = stablehlo.subtract %3, %22 : tensor<f32>
    %24 = stablehlo.log %23 : tensor<f32>
    %25 = stablehlo.multiply %19, %24 : tensor<f32>
    %26 = stablehlo.multiply %1, %25 : tensor<f32>
    %27 = stablehlo.negate %26 : tensor<f32>
    %28 = stablehlo.add %0, %27 : tensor<f32>
    return %28, %4 : tensor<f32>, tensor<2xui64>
  }
}
