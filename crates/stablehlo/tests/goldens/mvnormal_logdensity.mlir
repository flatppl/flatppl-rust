module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<[0.2, 0.1]> : tensor<2xf32>
    %1 = stablehlo.cholesky %arg1, lower = true : tensor<2x2xf32>
    %2 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %3 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %4 = stablehlo.compare EQ, %2, %3 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %5 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %6 = stablehlo.select %4, %1, %5 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%6 init: %7) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %9 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %10 = stablehlo.reduce(%8 init: %9) applies stablehlo.minimum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %11 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %12 = stablehlo.reduce(%8 init: %11) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %13 = stablehlo.constant dense<0.0> : tensor<f32>
    %14 = stablehlo.compare GT, %10, %13 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %15 = stablehlo.compare LT, %12, %9 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = stablehlo.select %14, %15, %14 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %17 = stablehlo.constant dense<0> : tensor<i32>
    %18 = stablehlo.constant dense<9> : tensor<i32>
    %19 = stablehlo.select %16, %17, %18 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %20 = stablehlo.log %8 : tensor<2xf32>
    %21 = stablehlo.reduce(%20 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %22 = stablehlo.constant dense<2.0> : tensor<f32>
    %23 = stablehlo.multiply %22, %21 : tensor<f32>
    %24 = stablehlo.constant dense<-0.5> : tensor<f32>
    %25 = stablehlo.multiply %24, %23 : tensor<f32>
    %26 = stablehlo.subtract %0, %arg0 : tensor<2xf32>
    %27 = stablehlo.reshape %26 : (tensor<2xf32>) -> tensor<1x2xf32>
    %28 = "stablehlo.triangular_solve"(%1, %27) <{left_side = false, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>}> : (tensor<2x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    %29 = stablehlo.reshape %28 : (tensor<1x2xf32>) -> tensor<2xf32>
    %30 = stablehlo.multiply %29, %29 : tensor<2xf32>
    %31 = stablehlo.reduce(%30 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %32 = stablehlo.multiply %24, %31 : tensor<f32>
    %33 = stablehlo.constant dense<-1.8378770664093453> : tensor<f32>
    %34 = stablehlo.add %33, %25 : tensor<f32>
    %35 = stablehlo.add %34, %32 : tensor<f32>
    return %35, %19 : tensor<f32>, tensor<i32>
  }
}
