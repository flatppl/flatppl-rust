module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<i32>, %arg1: tensor<3xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.compare LT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %4 = stablehlo.reduce(%arg1 init: %3) applies stablehlo.minimum across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.constant dense<0.0> : tensor<f32>
    %6 = stablehlo.compare LT, %4, %5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg1 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %9 = stablehlo.subtract %8, %1 : tensor<f32>
    %10 = stablehlo.abs %9 : tensor<f32>
    %11 = stablehlo.constant dense<0.000001430511474609375> : tensor<f32>
    %12 = stablehlo.compare GT, %10, %11 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %13 = stablehlo.select %6, %6, %12 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %14 = stablehlo.select %2, %2, %13 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %15 = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.constant dense<9> : tensor<i32>
    %17 = stablehlo.select %14, %16, %15 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %19 = stablehlo.add %0, %1 : tensor<f32>
    %20 = chlo.lgamma %19 : tensor<f32> -> tensor<f32>
    %22 = stablehlo.constant dense<"0x00000040000040400000A040"> : tensor<3xf32>
    %24 = stablehlo.constant dense<"0x1872313F6058E53F22339940"> : tensor<3xf32>
    %25 = stablehlo.reduce(%24 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %26 = stablehlo.negate %25 : tensor<f32>
    %27 = stablehlo.log %arg1 : tensor<3xf32>
    %28 = stablehlo.multiply %22, %27 : tensor<3xf32>
    %29 = stablehlo.reduce(%28 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %30 = stablehlo.add %20, %26 : tensor<f32>
    %31 = stablehlo.add %30, %29 : tensor<f32>
    return %31, %17 : tensor<f32>, tensor<i32>
  }
}
