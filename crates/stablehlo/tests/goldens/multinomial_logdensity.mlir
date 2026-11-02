module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<i32>, %arg1: tensor<3xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.compare LT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.abs %arg1 : tensor<3xf32>
    %5 = stablehlo.subtract %arg1, %4 : tensor<3xf32>
    %6 = stablehlo.reshape %arg1 : (tensor<3xf32>) -> tensor<1x3xf32>
    %7 = stablehlo.reshape %5 : (tensor<3xf32>) -> tensor<1x3xf32>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%8 init: %9) applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
    %11 = stablehlo.slice %10 [0:1] : (tensor<2xf32>) -> tensor<1xf32>
    %12 = stablehlo.reshape %11 : (tensor<1xf32>) -> tensor<f32>
    %13 = stablehlo.slice %10 [1:2] : (tensor<2xf32>) -> tensor<1xf32>
    %14 = stablehlo.reshape %13 : (tensor<1xf32>) -> tensor<f32>
    %15 = stablehlo.compare LT, %14, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = stablehlo.subtract %12, %1 : tensor<f32>
    %17 = stablehlo.abs %16 : tensor<f32>
    %18 = stablehlo.maximum %12, %1 : tensor<f32>
    %19 = stablehlo.constant dense<0.00000286102294921875> : tensor<f32>
    %20 = stablehlo.multiply %18, %19 : tensor<f32>
    %21 = stablehlo.compare GT, %17, %20 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %22 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %23 = stablehlo.compare EQ, %12, %22 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %24 = stablehlo.select %15, %15, %21 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %25 = stablehlo.select %24, %24, %23 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %26 = stablehlo.select %2, %2, %25 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %27 = stablehlo.constant dense<0> : tensor<i32>
    %28 = stablehlo.constant dense<9> : tensor<i32>
    %29 = stablehlo.select %26, %28, %27 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %31 = stablehlo.add %0, %1 : tensor<f32>
    %32 = chlo.lgamma %31 : tensor<f32> -> tensor<f32>
    %34 = stablehlo.constant dense<"0x00000040000040400000A040"> : tensor<3xf32>
    %36 = stablehlo.constant dense<"0x1872313F6058E53F22339940"> : tensor<3xf32>
    %37 = stablehlo.reduce(%36 init: %9) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %38 = stablehlo.negate %37 : tensor<f32>
    %39 = stablehlo.log %arg1 : tensor<3xf32>
    %40 = stablehlo.multiply %34, %39 : tensor<3xf32>
    %41 = stablehlo.reduce(%40 init: %9) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %42 = stablehlo.add %32, %38 : tensor<f32>
    %43 = stablehlo.add %42, %41 : tensor<f32>
    return %43, %29 : tensor<f32>, tensor<i32>
  }
}
