module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<i32>, %arg1: tensor<3xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.compare LT, %arg0, %0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %4 = stablehlo.compare LT, %arg1, %3 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%arg1 init: %5) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.subtract %6, %7 : tensor<f32>
    %9 = stablehlo.abs %8 : tensor<f32>
    %10 = stablehlo.constant dense<0.000001430511474609375> : tensor<f32>
    %11 = stablehlo.compare GT, %9, %10 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = stablehlo.constant dense<false> : tensor<i1>
    %13 = stablehlo.reduce(%4 init: %12) applies stablehlo.or across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
    %14 = stablehlo.select %13, %13, %11 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %15 = stablehlo.select %1, %1, %14 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %16 = stablehlo.constant dense<0> : tensor<i32>
    %17 = stablehlo.constant dense<9> : tensor<i32>
    %18 = stablehlo.select %15, %17, %16 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %20 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<f32>
    %21 = stablehlo.add %20, %7 : tensor<f32>
    %22 = chlo.lgamma %21 : tensor<f32> -> tensor<f32>
    %24 = stablehlo.constant dense<"0x00000040000040400000A040"> : tensor<3xf32>
    %26 = stablehlo.constant dense<"0x1872313F6058E53F22339940"> : tensor<3xf32>
    %27 = stablehlo.reduce(%26 init: %5) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %28 = stablehlo.negate %27 : tensor<f32>
    %29 = stablehlo.log %arg1 : tensor<3xf32>
    %30 = stablehlo.multiply %24, %29 : tensor<3xf32>
    %31 = stablehlo.reduce(%30 init: %5) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %32 = stablehlo.add %22, %28 : tensor<f32>
    %33 = stablehlo.add %32, %31 : tensor<f32>
    return %33, %18 : tensor<f32>, tensor<i32>
  }
}
